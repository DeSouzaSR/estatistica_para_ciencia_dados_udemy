# O objetivo deste código é implementar a seleção de uma amostra 
# sistemaática 

import pandas as pd
pd.set_option('display.max_columns', 500)
import random
import numpy as np

import sys
sys.path.append('./src')
from amostragem import amostragem_sistematica

# Importando base de dados do censo dos Estados Unidos
# O objetivo deste dataset, é ser usado para aprendizado de máquina para
# prever se a renda pessoa será maior ou menor que 50000 dolares.
dataset = pd.read_csv('data/raw_data/census.csv', sep=',', encoding='utf8')

df_amostra_sistematica = amostragem_sistematica(dataset, 100)
