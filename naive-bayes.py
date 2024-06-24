"""
Fazer análise Naïves Bayes para dados de cessão de crédito bancário
"""
# Importando os dados
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Lendo dataset
dataset = pd.read_csv('data/raw_data/credit_data.csv')
print("\nArquivo lido com sucesso")
print(f"Dataset tem {dataset.shape[0]} linhas e {dataset.shape[1]} colunas")

# Removendo valores faltantes
print("\nRemovendo dados faltantes...")
dataset.dropna(inplace=True)
print(f"Dataset tem {dataset.shape[0]} linhas e {dataset.shape[1]} colunas")

# Verificando que os dados estão desbalanceados
# Plot contagem de "c#default"
# print("Plot para verificar dados desbalanceados")
# sns.countplot(dataset, x='c#default')
# plt.show()

# Separar atributos previsores x e a classe y
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:,4]

# Dividindo treinamento e teste
# A seleção está estratificada, por causa do desbalanceamento da base de dados
# O valor 0.2 significa 20% da base para fazer o teste. 
# stratify = y significa que usará os valores de y como referência.
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,\
     test_size = 0.2, stratify = y)
     
# Treinamento do modelo
print("\nTreinando modelo...")
modelo = GaussianNB()

# Ajustando o modelo treinado aos nossos dados
modelo.fit(X_treinamento, y_treinamento)

# Privisões
print("\nCalculando previsões com o teste...")
previsoes = modelo.predict(X_teste)

# Verificando acurácia
print("\nVerificando acurácia...")
print(f"Acurácia: {accuracy_score(previsoes, y_teste)}")

# Matriz de confusão
cm = confusion_matrix(previsoes, y_teste)
sns.heatmap(cm, annot=True)
plt.show()

# Finalizando
print("\nAnálise finalizada com sucesso.")
