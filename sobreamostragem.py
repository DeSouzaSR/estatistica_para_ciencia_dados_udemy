

# Aplicar subamostragem com a técnica Tomke linnks
# Instalação
# pip install imbalanced-learn
import pandas as pd
import numpy as np
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
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

print("\nExecutando a sobreamostragem")
smote = SMOTE(sampling_strategy='minority')
X_over, y_over = smote.fit_resample(X, y)

print("\nTreinamento e teste")
X_treinamento_o, X_teste_o, y_treinamento_o, y_teste_o = \
	train_test_split(X_over, y_over, test_size=0.2, stratify=y_over)					  
print(X_treinamento_o.shape, X_teste_o.shape)

# Gaussian
modelo_o = GaussianNB()
modelo_o.fit(X_treinamento_o, y_treinamento_o)
previsoes_o = modelo_o.predict(X_teste_o)

print("\nAccurácia:")
print(accuracy_score(previsoes_o, y_teste_o))

print("\nMatriz de Confusão")
cm = confusion_matrix(previsoes_o, y_teste_o)
print(cm)
