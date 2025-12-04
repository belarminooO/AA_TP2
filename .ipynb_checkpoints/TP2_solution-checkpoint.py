import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix
from sklearn.cluster import KMeans
import re

def carregar_dados(caminho_ficheiro="imdbFull.p"):
    print("A carregar dados...")
    with open(caminho_ficheiro, 'rb') as f:
        dados = pickle.load(f)
    return dados.data, dados.target

def pre_processar_texto(texto):
    # Pré-processamento simples: remover tags HTML, manter apenas letras, minúsculas
    texto = re.sub(r'<br\s*/?>', ' ', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    return texto.lower()

def extrair_features(textos, max_features=5000):
    print("A vetorizar texto...")
    # Usando TfidfVectorizer como pedido
    vetorizador = TfidfVectorizer(
        preprocessor=pre_processar_texto,
        stop_words='english',
        max_features=max_features,
        ngram_range=(1, 2) # Unigramas e bigramas
    )
    X = vetorizador.fit_transform(textos)
    return X, vetorizador

def tarefa_classificacao(X_treino, X_teste, y_treino, y_teste):
    print("\n--- Tarefa 1: Classificação ---")
    classificador = LogisticRegression(max_iter=1000, random_state=42)
    classificador.fit(X_treino, y_treino)
    y_pred = classificador.predict(X_teste)
    
    print(f"Acurácia: {accuracy_score(y_teste, y_pred):.4f}")
    print("Relatório de Classificação:")
    print(classification_report(y_teste, y_pred))
    return classificador

def tarefa_regressao(X_treino, X_teste, y_treino, y_teste):
    print("\n--- Tarefa 2: Regressão ---")
    regressor = Ridge(alpha=1.0, random_state=42)
    regressor.fit(X_treino, y_treino)
    y_pred_bruto = regressor.predict(X_teste)
    
    # Converter saída da regressão para classes (1-4, 7-10)
    # Arredondar e limitar ao intervalo [1, 10]
    
    y_pred_arredondado = np.round(y_pred_bruto)
    y_pred_arredondado = np.clip(y_pred_arredondado, 1, 10)
    
    # Calcular MSE nas previsões brutas
    mse = mean_squared_error(y_teste, y_pred_bruto)
    print(f"MSE (bruto): {mse:.4f}")
    
    # Calcular Acurácia nas previsões arredondadas
    acc = accuracy_score(y_teste, y_pred_arredondado)
    print(f"Acurácia (arredondada): {acc:.4f}")
    
    return regressor

def tarefa_clustering(X, vetorizador, n_clusters=2):
    print("\n--- Tarefa 3: Clustering ---")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X)
    
    print("Termos principais por cluster:")
    centroides_ordenados = kmeans.cluster_centers_.argsort()[:, ::-1]
    termos = vetorizador.get_feature_names_out()
    for i in range(n_clusters):
        print(f"Cluster {i}: ", end='')
        for ind in centroides_ordenados[i, :10]:
            print(f'{termos[ind]} ', end='')
        print()
    return kmeans

def main():
    # 1. Carregar Dados
    textos, classes = carregar_dados()
    
    # 2. Pré-processamento e Extração de Features
    X, vetorizador = extrair_features(textos)
    
    # Divisão dos dados
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, classes, test_size=0.3, random_state=42)
    
    # 3. Classificação
    tarefa_classificacao(X_treino, X_teste, y_treino, y_teste)
    
    # 4. Regressão
    tarefa_regressao(X_treino, X_teste, y_treino, y_teste)
    
    # 5. Clustering
    tarefa_clustering(X[:10000], vetorizador, n_clusters=2)

if __name__ == "__main__":
    main()
