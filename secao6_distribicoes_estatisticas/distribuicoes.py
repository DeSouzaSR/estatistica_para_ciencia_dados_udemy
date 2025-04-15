# %% [markdown]
# ---
# title: Palmer Penguins
# author: Norah Jones
# date: 3/12/23
# ---

# Distribuições estatísitcas

# %%
# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# %%
# Variáveis contínuas
## Distribuição normal
dados_normal = stats.norm.rvs(loc=0, scale=1, size=1000)  # Gera 1000 dados de uma normal padrão
print('Distribuição normal')
print('minimo:', dados_normal.min())
print('maximo:', dados_normal.max())

# %%
# Plotar usando seaborn
sns.distplot(dados_normal, kde=True)
plt.title('Distribuição Normal')
plt.xlabel('Valores')
plt.ylabel('Frequência')
plt.show()


# %%
print("Valores da Média, Mediana, Moda, Variância e Desvio Padrão")
print('-'*80)
print("Média:", dados_normal.mean())
print("Mediana:", np.median(dados_normal))
print("Moda:", stats.mode(dados_normal)[0])
print("Variância:", dados_normal.var())
print("Desvio Padrão:", dados_normal.std())

# %%
