import pandas as pd
pd.set_option('display.max_columns', 500)
import random
import numpy as np


# Functions
def simple_random_sample(dataset, samples):
    """ Return a simple random sample """
    return dataset.sample(n = samples, random_state=314)


def amostragem_sistematica(dataset, amostras):
    """Returna uma amostra sistemática"""
    intervalo = len(dataset) // amostras
    random.seed(1)
    inicio = random.randint(0, intervalo)
    indices = np.arange(inicio, len(dataset), step=intervalo)
    amostra_sistematica = dataset.iloc[indices]
    return amostra_sistematica

def amostragem_agrupamento(dataset, numero_grupos):
    intervalo = len(dataset) // numero_grupos
    grupos = []
    id_grupo = 0
    contagem = 0

    for _ in dataset.iterrows():
        grupos.append(id_grupo)
        contagem += 1
        if contagem > intervalo:
            contagem = 0
            id_grupo += 1

    dataset['grupo'] = grupos
    random.seed(1)
    grupo_selecionado = random.randint(0, numero_grupos)
    return dataset[dataset['grupo'] == grupo_selecionado]


# Input data
path_file = r"../data/raw_data/census.csv"
dataset = pd.read_csv(path_file, sep=',', encoding='utf8')
print("Base de dados")
print(f'Dimensão: {dataset.shape[0]} linhas e {dataset.shape[1]} colunas')
#print(dataset.sample(n=5))

# Amostra sistemática
df_amostra_sistematica = amostragem_sistematica(dataset, 100)

print("\nAmostra sistemática")
print(f'Dimensão: {df_amostra_sistematica.shape[0]} linhas e {df_amostra_sistematica.shape[1]} colunas')

# Amostra por grupos
df_amostra_agrupamento = amostragem_agrupamento(dataset, 100)
print("\nAmostra agrupamento")
print(f'Dimensão: {df_amostra_agrupamento.shape[0]} linhas e {df_amostra_agrupamento.shape[1]} colunas')

