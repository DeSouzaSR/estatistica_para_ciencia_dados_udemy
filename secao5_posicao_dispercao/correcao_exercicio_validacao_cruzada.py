# Importar módulos
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Importar dataset
path_file = r'../data/raw_data/credit_data.csv'
dataset = pd.read_csv(path_file)

dataset.dropna(inplace=True)

# Variáveis preditoras
X = dataset.iloc[:, 1:4].values

# Variável que será prevista
y = dataset.iloc[:,4].values

resultados_naive_bayes_cv = []
resultados_logistica_cv = []
resultados_forest_cv = []
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    naive_bayes = GaussianNB()
    scores = cross_val_score(naive_bayes, X, y, cv = kfold)
    resultados_naive_bayes_cv.append(scores.mean())

    logistica = LogisticRegression()
    scores = cross_val_score(logistica, X, y, cv = kfold)
    resultados_logistica_cv.append(scores.mean())

    random_forest = RandomForestClassifier()
    scores = cross_val_score(random_forest, X, y, cv = kfold)
    resultados_forest_cv.append(scores.mean())


# Resultados
print(resultados_naive_bayes_cv)
print(resultados_logistica_cv)
print(resultados_forest_cv)

# Resultados do coeficiente de variação
print("Coeficiente de variação")
print(
    stats.variation(resultados_naive_bayes_cv) * 100,
    stats.variation(resultados_logistica_cv) * 100,
    stats.variation(resultados_forest_cv) * 100
)