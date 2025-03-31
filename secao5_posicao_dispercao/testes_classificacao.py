#%%
# Importar módulos
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# %%
# Importar dataset
path_file = r'../data/raw_data/credit_data.csv'
dataset = pd.read_csv(path_file)
# %%
dataset.dropna(inplace=True)
# %%
# Variáveis preditoras
X = dataset.iloc[:, 1:4].values
# %%
# Variável que será prevista
y = dataset.iloc[:,4].values
# %%
# Elaborar 30 testes
resultados_naive_bayes = []
resultados_logistica = []
resultados_forest = []
for i in range(30): # Numero aceito pela comunidade
    X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=i
    )
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_treinamento, y_treinamento)
    resultados_naive_bayes.append(accuracy_score(y_teste, naive_bayes.predict(X_teste)))

    logistica = LogisticRegression()
    logistica.fit(X_treinamento, y_treinamento)
    resultados_logistica.append(accuracy_score(y_teste, logistica.predict(X_teste)))

    random_forest = RandomForestClassifier()
    random_forest.fit(X_treinamento, y_treinamento)
    resultados_forest.append(accuracy_score(y_teste, random_forest.predict(X_teste)))
# %%
# Convertendo para array numpy
resultados_naive_bayes = np.array(resultados_naive_bayes)
resultados_logistica = np.array(resultados_logistica)
resultados_forest = np.array(resultados_forest)
# %%
# Média
resultados_naive_bayes.mean(), resultados_logistica.mean(), resultados_forest.mean()

# %%
# Moda
stats.mode(resultados_naive_bayes), stats.mode(resultados_logistica), stats.mode(resultados_forest)

# %%
np.median(resultados_naive_bayes), np.median(resultados_logistica), np.median(resultados_forest)
# %%
# Variância
np.var(resultados_naive_bayes), np.var(resultados_logistica), np.var(resultados_forest)
# %%
# Desvio padrão
np.std(resultados_naive_bayes), np.std(resultados_logistica), np.std(resultados_forest)
# %%
# Coeficiente de variação
stats.variation(resultados_naive_bayes) *100, stats.variation(resultados_logistica) *100, stats.variation(resultados_forest) *100
# %%
