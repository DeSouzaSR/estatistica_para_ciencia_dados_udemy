import pandas as pd
pd.set_option('display.max_columns', 500)
import random
import numpy as np


# Functions
def simple_random_sample(dataset, samples):
    """ Return a simple random sample """
    return dataset.sample(n = samples, random_state=314)

# Input data
path_file = r"../data/raw_data/census.csv"
dataset = pd.read_csv(path_file, sep=',', encoding='utf8')
print("Base de dados")
print(f'Dimensão: {dataset.shape[0]} linhas e {dataset.shape[1]} colunas')
#print(dataset.sample(n=5))

# Simple random sample
df_simple_random_sample = simple_random_sample(dataset, 100)
print("Amostragem aleatória simples ")
print(f'Dimensão: {df_simple_random_sample.shape[0]} linhas e {df_simple_random_sample.shape[1]} colunas')
#print(df_simple_random_sample.sample(n=5))

print("\nPrograma concluído com sucesso!")