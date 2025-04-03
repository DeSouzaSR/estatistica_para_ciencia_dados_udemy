"""
O objetivo deste exercício é utilizar a base de dados do crédito e aplicar a
técnica de seleção de atributos utilizando variância
- Carregue o arquivo credit_data.csv
- Calcule a variância para os atributos income, age e loan e aplique o 
    método de seleção Low Variance
- Faça um teste do accuracy utilizando o algoritmo Naïve Bayes, sem seleção 
    de atributos e com seleção de atributos
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Read dataset
dataset = pd.read_csv('../data/raw_data/credit_data.csv')

# Remover NaN's
dataset.dropna(inplace=True)

# Separar atributos e variável alvo
X = dataset.iloc[:,1:4].values
y = dataset.iloc[:,4].values

# Normalização (os dados estão muito discrepantes)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Aplicando a técnica de seleção de atributo
# A menor variância é 0.027, obtido com np.var
selecao = VarianceThreshold(threshold=0.027)
X_novo = selecao.fit_transform(X)

# Comparando com e sem seleção de atributos
# Sem seleção
naive_sem_selecao = GaussianNB()
naive_sem_selecao.fit(X, y)
previsoes_sem_selecao = naive_sem_selecao.predict(X)

# Com seleção
naive_com_selecao = GaussianNB()
naive_com_selecao.fit(X_novo, y)
previsoes_com_selecao = naive_com_selecao.predict(X_novo)

# Resultados
print(f'Acurácia do dataset sem seleção: {accuracy_score(previsoes_sem_selecao, y)}')
print(f'Acurácia do dataset sem seleção: {accuracy_score(previsoes_com_selecao, y)}')

np.nan

