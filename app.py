from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from io import StringIO
import datetime

app = Flask(__name__)

# Inicializações (executar só uma vez)
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("pt_core_news_sm")
stop_words = set(stopwords.words('portuguese'))
custom_stopwords = {'app', 'aplicativo', 'aplicação', 'uso', 'usar', 'pra', 'vou'}
stop_words.update(custom_stopwords)
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def preprocess(text):
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-záéíóúãõâêîôç ]", "", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct and len(token.text) > 2]
    return " ".join(tokens)

def get_sentiment_bert(text):
    if not text or not isinstance(text, str) or text.isspace():
        return "neutral"
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item() + 1
        return "negative" if predicted_class <= 2 else "neutral" if predicted_class == 3 else "positive"
    except:
        return "neutral"

def extract_aspects(text):
    if not text:
        return []
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]
    relevant_bigrams = []
    for gram in bigrams:
        gram_doc = nlp(gram)
        if any(token.pos_ in ['NOUN', 'VERB'] for token in gram_doc) and not all(word in stop_words for word in gram.split()):
            relevant_bigrams.append(gram)
    aspects = nouns + relevant_bigrams
    return list(set(aspects))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html", error="Nenhum arquivo enviado.")
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template("index.html", error=f"Erro ao ler CSV: {e}")

        # Detectar coluna de comentários
        possible_names = ['review', 'comentario', 'text', 'comment', 'avaliacao', 'feedback']
        review_column = next((col for col in df.columns if col.lower() in possible_names), df.columns[0])

        df[review_column] = df[review_column].astype(str)
        df['cleaned'] = df[review_column].apply(preprocess)
        df['sentiment'] = df[review_column].apply(get_sentiment_bert)
        df['aspects'] = df['cleaned'].apply(extract_aspects)

        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)

        aspect_sentiments = defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0})
        aspect_examples = defaultdict(lambda: {'positive': [], 'neutral': [], 'negative': []})

        for _, row in df.iterrows():
            for aspect in row['aspects']:
                aspect_sentiments[aspect][row['sentiment']] += 1
                if len(aspect_examples[aspect][row['sentiment']]) < 3:
                    aspect_examples[aspect][row['sentiment']].append(row[review_column])

        MIN_MENTIONS = 3
        POSITIVE_THRESHOLD = 0.6
        NEGATIVE_THRESHOLD = 0.4

        filtered_aspects = {
            aspect: counts for aspect, counts in aspect_sentiments.items()
            if sum(counts.values()) >= MIN_MENTIONS
        }

        aspect_ratios = {
            aspect: {
                'positive_ratio': counts['positive'] / sum(counts.values()),
                'negative_ratio': counts['negative'] / sum(counts.values()),
                'total_mentions': sum(counts.values())
            }
            for aspect, counts in filtered_aspects.items()
        }

        strengths = [
            (aspect, ratios) for aspect, ratios in aspect_ratios.items()
            if ratios['positive_ratio'] >= POSITIVE_THRESHOLD
        ]
        strengths = sorted(strengths, key=lambda x: (-x[1]['positive_ratio'], -x[1]['total_mentions']))[:10]

        weaknesses = [
            (aspect, ratios) for aspect, ratios in aspect_ratios.items()
            if ratios['negative_ratio'] >= NEGATIVE_THRESHOLD
        ]
        weaknesses = sorted(weaknesses, key=lambda x: (-x[1]['negative_ratio'], -x[1]['total_mentions']))[:10]

        def format_examples(aspect, sentiment):
            examples = aspect_examples[aspect][sentiment]
            if not examples:
                return "<p><i>Nenhum exemplo disponível.</i></p>"
            return "<ul>" + "".join(f'<li>{ex}</li>' for ex in examples) + "</ul>"

        positive_pct = sentiment_counts.get('positive', 0) / total * 100
        neutral_pct = sentiment_counts.get('neutral', 0) / total * 100
        negative_pct = sentiment_counts.get('negative', 0) / total * 100

        # Passar dados para o template
        context = {
            'positive_count': sentiment_counts.get('positive', 0),
            'neutral_count': sentiment_counts.get('neutral', 0),
            'negative_count': sentiment_counts.get('negative', 0),
            'positive_pct': positive_pct,
            'neutral_pct': neutral_pct,
            'negative_pct': negative_pct,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'format_examples': format_examples,
            'pd_timestamp': datetime.datetime.now().strftime('%d/%m/%Y %H:%M'),
            'MIN_MENTIONS': MIN_MENTIONS,
            'POSITIVE_THRESHOLD': int(POSITIVE_THRESHOLD*100),
            'NEGATIVE_THRESHOLD': int(NEGATIVE_THRESHOLD*100),
        }

        return render_template("report.html", **context)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
