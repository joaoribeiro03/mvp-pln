Bruno Monteiro, 
🧠 Análise de Sentimentos e Aspectos em Comentários
Este projeto realiza uma análise automatizada de sentimentos e extração de aspectos em comentários em português, gerando um relatório interativo em HTML. Ele utiliza transformers, spaCy, e NLTK para fornecer insights sobre os pontos fortes e fracos percebidos nos comentários.


📦 Funcionalidades
Classificação de sentimentos (positivo, neutro, negativo) usando BERT Multilingual.

Limpeza e lematização de texto em português.

Extração de aspectos (substantivos e bigramas significativos).

Relatório em HTML com:

Gráfico de distribuição de sentimentos.

Principais pontos fortes e fracos com exemplos reais.


🚀 Como Usar
1. Abrir no Google Colab
Recomendado: Execute o código diretamente no Google Colab.

2. Instale as dependências
No início do notebook, as bibliotecas necessárias são instaladas automaticamente:

python
Copiar
Editar
!pip install -q pandas numpy matplotlib seaborn nltk spacy scikit-learn transformers torch
!python -m spacy download pt_core_news_sm -q
3. Faça upload do arquivo CSV
O sistema solicitará o upload de um arquivo CSV contendo os comentários.

O arquivo deve conter uma coluna de texto, como por exemplo: comentario, review, feedback, etc.

4. Execute todas as células
O código irá:

Limpar os textos.

Classificar o sentimento com BERT.

Extrair aspectos dos textos.

Gerar um relatório em HTML automaticamente, com download disponível.


📊 Exemplo de Saída
O relatório gerado inclui:

Gráfico de distribuição de sentimentos

Lista de pontos fortes (aspectos com avaliações majoritariamente positivas)

Lista de áreas para melhoria (aspectos com avaliações majoritariamente negativas)

Exemplos reais de comentários para cada aspecto


🧠 Modelos Utilizados
Modelo de Sentimentos: nlptown/bert-base-multilingual-uncased-sentiment

spaCy: pt_core_news_sm (modelo de linguagem para português)

NLTK: Stopwords, tokenização e n-gramas

🗂 Requisitos do Arquivo CSV
Deve conter uma coluna com os comentários.

O nome da coluna pode ser comentario, review, text, feedback, etc. O script tentará detectar automaticamente.


💡 Sugestões
Use arquivos com comentários reais de clientes para obter insights úteis para negócios, produtos, ou serviços.

Os resultados são mais eficazes com um número razoável de comentários (mínimo recomendado: 30).


📁 Saída
Arquivo gerado: relatorio_analise_comentarios.html

Faça o download ao final do processo para visualizar localmente.


🧪 Exemplo de Execução
Execute as células no notebook.

Faça upload de um arquivo como:

csv
Copiar
Editar
comentario
"Ótimo atendimento, muito rápido."
"Aplicativo trava demais."
"Preço justo e fácil de usar."
Visualize e baixe o relatório final.


⚠️ Observações
O modelo de sentimentos não é específico para o português, mas oferece resultados razoáveis com comentários simples.

Comentários muito curtos ou com erros gramaticais podem ter classificação imprecisa.


📄 Licença
Este projeto é de uso educacional e experimental.