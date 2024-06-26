
# Aplicar subamostragem com a técnica Tomke linnks
# Instalação
# pip install imbalanced-learn
import pandas as pd
import numpy as np
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

print("\nLendo dataset...")
dataset = pd.read_csv('data/raw_data/credit_data.csv')

print("\nClear NAs")
dataset.dropna(inplace=True)

print("\nSeparando previsores...")
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:,4]

print("\nSeparando subamostras...")
tl = TomekLinks(sampling_strategy='majority')
X_under, y_under = tl.fit_resample(X, y) 

print("\nTreinamento e teste")
X_treinamento_u, X_teste_u, y_treinamento_u, y_teste_u = train_test_split(X_under,
																		  y_under,
																		  test_size=0.2,

																	  stratify=y_under)					  
print(X_treinamento_u.shape, X_teste_u.shape)

# Gaussian
modelo_u = GaussianNB()
modelo_u.fit(X_treinamento_u, y_treinamento_u)
previsoes_u = modelo_u.predict(X_teste_u)

print("\nAccurácia:")
print(accuracy_score(previsoes_u, y_teste_u))

print("\nMatriz de Confusão")
cm = confusion_matrix(previsoes_u, y_teste_u)
print(cm)
