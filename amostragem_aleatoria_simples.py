import pandas as pd
pd.set_option('display.max_columns', 500)
import random
import numpy as np

import sys
sys.path.append('./src')
from amostragem import amostragem_aleatoria_simples


# Importando base de dados do censo dos Estados Unidos
# O objetivo deste dataset, é ser usado para aprendizado de máquina para
# prever se a renda pessoa será maior ou menor que 50000 dolares.
dataset = pd.read_csv('data/raw_data/census.csv', sep=',', encoding='utf8')
print("Base de dados")
print(f'Dimensão: {dataset.shape[0]} linhas e {dataset.shape[1]} colunas')
print(dataset.sample(n=5))

# Selecionando uma amostra aleatória
# random_state é equivalente ao seed, e garante a reprodutibilidade
df_amostra_aleatoria_simples = amostragem_aleatoria_simples(dataset, 100)
print("Amostragem ")
print(f'Dimensão: {df_amostra_aleatoria_simples.shape[0]} linhas e {df_amostra_aleatoria_simples.shape[1]} colunas')
print(df_amostra_aleatoria_simples.sample(n=5))
