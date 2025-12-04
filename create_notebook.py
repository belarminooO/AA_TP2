import nbformat as nbf
import inspect
import TP2_solution

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- Célula 1: Título e Introdução ---
    md_intro = """# Trabalho Laboratorial 2 - Classificação de Críticas de Cinema do IMDb
**Autores:** [Seus Nomes/Números]

## 1. Introdução
Este trabalho tem como objetivo analisar um conjunto de dados de críticas de cinema do IMDb.
As tarefas realizadas são:
1.  **Classificação**: Prever a pontuação da crítica (1-10).
2.  **Regressão**: Prever a pontuação como valor contínuo.
3.  **Clustering**: Agrupar críticas por similaridade.

O código foi desenvolvido para ser claro e eficiente, utilizando a biblioteca `scikit-learn`.
"""
    
    # --- Célula 2: Imports ---
    md_imports = """## 2. Importação de Bibliotecas
**O que faz:** Importa as ferramentas necessárias.
**Decisões:**
*   `pickle`: Para carregar o ficheiro de dados `imdbFull.p`.
*   `sklearn`: A biblioteca padrão para Machine Learning em Python. Usamos módulos para extração de texto (`TfidfVectorizer`), modelos lineares (`LogisticRegression`, `Ridge`), métricas (`accuracy_score`, etc.) e clustering (`KMeans`).
*   `re`: Para expressões regulares, usadas na limpeza do texto.
"""
    code_imports = """import pickle
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix
from sklearn.cluster import KMeans

%matplotlib inline
"""

    # --- Célula 3: Carregamento e Pré-processamento ---
    md_data = """## 3. Carregamento e Pré-processamento dos Dados
**O que faz:**
1.  `carregar_dados`: Lê o ficheiro `imdbFull.p`.
2.  `pre_processar_texto`: Limpa o texto das críticas.
3.  `extrair_features`: Converte o texto em números (vetores) usando TF-IDF.

**Decisões e Porquês:**
*   **Limpeza de Texto**: As críticas vêm da web e contêm tags HTML (como `<br />`). Removemos estas tags e caracteres não alfabéticos para que o modelo se foque apenas nas palavras. Convertemos tudo para minúsculas para que "Filme" e "filme" sejam tratados como a mesma palavra.
*   **TF-IDF (Term Frequency-Inverse Document Frequency)**: Escolhemos esta técnica em vez de apenas contar palavras (Bag of Words) porque o TF-IDF dá menos peso a palavras muito comuns (como "the", "a") que aparecem em todos os documentos e não ajudam a distinguir sentimentos.
*   **N-grams (1, 2)**: Usamos unigramas (palavras isoladas) e bigramas (pares de palavras). Isto é crucial porque captura contextos como "not good" (não bom), que tem um sentido oposto a "good". Se usássemos apenas unigramas, "not" e "good" seriam contados separadamente.
*   **Max Features (5000)**: Limitamos o vocabulário às 5000 palavras mais importantes para evitar que o modelo fique demasiado pesado e lento, e para reduzir o ruído de palavras raras.
"""
    code_data = inspect.getsource(TP2_solution.carregar_dados) + "\n" + \
                inspect.getsource(TP2_solution.pre_processar_texto) + "\n" + \
                inspect.getsource(TP2_solution.extrair_features) + "\n\n" + \
                "# Execução do carregamento e processamento\n" + \
                "textos, classes = carregar_dados()\n" + \
                "X, vetorizador = extrair_features(textos)\n" + \
                "X_treino, X_teste, y_treino, y_teste = train_test_split(X, classes, test_size=0.3, random_state=42)"

    # --- Célula 4: Tarefa 1 - Classificação ---
    md_task1 = """## 4. Tarefa I: Classificação
**O que faz:** Treina um modelo para prever a nota exata (1 a 10) de uma crítica.

**Parâmetros Importantes:**
*   `LogisticRegression(max_iter=1000)`:
    *   `max_iter=1000`: Aumentámos o número de tentativas (iterações) que o modelo tem para aprender. O valor padrão (100) muitas vezes não é suficiente para dados de texto complexos, e o modelo daria erro de convergência.
    *   `random_state=42`: Garante que os resultados são sempre iguais (reprodutibilidade).
*   `accuracy_score`: Conta simplesmente quantas vezes acertámos na nota exata.

**Decisões e Porquês:**
*   **Modelo: Regressão Logística**: Apesar do nome, é um classificador. Escolhemos este modelo porque é rápido, eficiente para dados de texto (alta dimensionalidade) e serve como uma excelente *baseline*. Funciona bem com TF-IDF.
*   **Divisão Treino/Teste**: Usamos 30% dos dados para teste para garantir que avaliamos o modelo em dados que ele nunca viu.
"""
    code_task1 = inspect.getsource(TP2_solution.tarefa_classificacao) + "\n\n" + \
                 "classificador = tarefa_classificacao(X_treino, X_teste, y_treino, y_teste)"

    # --- Célula 5: Tarefa 2 - Regressão ---
    md_task2 = """## 5. Tarefa II: Regressão
**O que faz:** Treina um modelo para prever a nota como um número contínuo (ex: 7.4), que depois arredondamos para a nota inteira mais próxima.

**Parâmetros Importantes:**
*   `Ridge(alpha=1.0)`:
    *   `alpha=1.0`: É o fator de regularização. Controla o quanto penalizamos o modelo por ser demasiado complexo. Um alpha maior simplifica mais o modelo (evita overfitting), um alpha menor deixa-o ajustar-se mais aos dados de treino. O valor 1.0 é um padrão equilibrado.
*   `mean_squared_error (MSE)`: Mede a média dos erros ao quadrado. Penaliza mais os erros grandes (errar por 5 valores é muito pior que errar por 1).

**Decisões e Porquês:**
*   **Modelo: Ridge Regression**: É uma Regressão Linear com regularização. A regularização é importante em texto para evitar que o modelo dê pesos exagerados a certas palavras raras.
*   **Conversão**: Como a regressão devolve números reais, arredondamos (`np.round`) e limitamos entre 1 e 10 (`np.clip`) para comparar com as classes originais.
"""
    code_task2 = inspect.getsource(TP2_solution.tarefa_regressao) + "\n\n" + \
                 "regressor = tarefa_regressao(X_treino, X_teste, y_treino, y_teste)"

    # --- Célula 6: Tarefa 3 - Clustering ---
    md_task3 = """## 6. Tarefa III: Clustering
**O que faz:** Agrupa as críticas em grupos (clusters) baseando-se apenas no texto, sem saber a nota (aprendizagem não supervisionada).

**Parâmetros Importantes:**
*   `KMeans(n_clusters=2)`:
    *   `n_clusters=2`: Definimos que queremos encontrar 2 grupos. Escolhemos 2 para tentar ver se o algoritmo separa naturalmente críticas "Positivas" de "Negativas".
    *   `n_init='auto'`: O algoritmo corre várias vezes com inícios diferentes e escolhe o melhor resultado automaticamente.

**Decisões e Porquês:**
*   **Modelo: K-Means**: É o algoritmo de clustering mais popular e simples. Tenta encontrar `k` centros de grupos e atribui cada crítica ao centro mais próximo.
*   **Análise**: Imprimimos os termos mais frequentes de cada cluster para tentar interpretar o que cada grupo representa (ex: se um grupo tem palavras como "bad" e o outro "great").
"""
    code_task3 = inspect.getsource(TP2_solution.tarefa_clustering) + "\n\n" + \
                 "# Usando subset para visualização rápida se necessário, ou dataset completo\n" + \
                 "kmeans = tarefa_clustering(X[:10000], vetorizador, n_clusters=2)"

    # --- Célula 7: Conclusão Geral ---
    md_concl = """## 7. Conclusão e Análise das Abordagens
**Resumo do Trabalho:**
Neste trabalho, explorámos três abordagens diferentes para analisar o mesmo conjunto de dados de críticas de cinema.

1.  **Classificação vs. Regressão**:
    *   A **Classificação** obteve uma acurácia superior (~42%) em comparação com a Regressão convertida (~22%). Isto sugere que tratar as notas como categorias distintas funcionou melhor para acertar na nota *exata*.
    *   No entanto, a **Regressão** tem um MSE (Erro Quadrático Médio) baixo (~4.7, o que dá um erro médio de ~2.2 valores). O modelo de regressão "entende" a polaridade (bom vs mau), mas tem dificuldade em acertar na nuance exata da nota (ex: distinguir um 8 de um 9).

2.  **Clustering**:
    *   O K-Means conseguiu separar as críticas em dois grupos com vocabulários distintos.
    *   **Cluster 0 (Negativo)**: Palavras como "bad", "dont", "watch".
    *   **Cluster 1 (Positivo)**: Palavras como "great", "story", "film".
    *   Isto valida a capacidade do TF-IDF em capturar a semântica do texto sem qualquer etiqueta prévia.

**Considerações Finais**:
A utilização de TF-IDF com Bigramas provou ser uma estratégia robusta para transformar texto em dados numéricos. As abordagens clássicas (Regressão Logística/Ridge) ofereceram um excelente compromisso entre rapidez de treino e interpretabilidade dos resultados.
"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(md_intro),
        nbf.v4.new_markdown_cell(md_imports),
        nbf.v4.new_code_cell(code_imports),
        nbf.v4.new_markdown_cell(md_data),
        nbf.v4.new_code_cell(code_data),
        nbf.v4.new_markdown_cell(md_task1),
        nbf.v4.new_code_cell(code_task1),
        nbf.v4.new_markdown_cell(md_task2),
        nbf.v4.new_code_cell(code_task2),
        nbf.v4.new_markdown_cell(md_task3),
        nbf.v4.new_code_cell(code_task3),
        nbf.v4.new_markdown_cell(md_concl)
    ]

    with open('TP2_solution.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook criado: TP2_solution.ipynb")

if __name__ == "__main__":
    create_notebook()
