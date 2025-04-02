import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from 

# Leitura dos dataset e separação nas variáveis preditoras 
# e nas variável alvo
X, y = datasets.load_iris(return_X_y=True)

# Separando treino e teste em 60/40 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0
)

# Classificador
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

# Validação cruzada
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Validação cruzada usando outra métrica
scores = cross_val_score(
    clf, X, y, cv=5, scoring='f1_macro'
)

# Usando cross validation para multiplas métricas
