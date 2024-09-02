import pandas as pd
pd.set_option('display.max_columns', 500)
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import sys
sys.path.append('./src')
#from amostragem import amostragem_agrupamento

# read dataset
dataset = pd.read_csv('data/raw_data/census.csv', sep=',', encoding='utf8')

# Take sample
df_amostra_estratificada = amostragem_estratificada(dataset, 100)