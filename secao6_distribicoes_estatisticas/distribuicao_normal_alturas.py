import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# alturas
dados = np.array([126. , 129.5, 133. , 133. , 136.5, 136.5, 140. , 140. , 140. ,
                  140. , 143.5, 143.5, 143.5, 143.5, 143.5, 143.5, 147. , 147. ,
                  147. , 147. , 147. , 147. , 147. , 150.5, 150.5, 150.5, 150.5,
                  150.5, 150.5, 150.5, 150.5, 154. , 154. , 154. , 154. , 154. ,
                  154. , 154. , 154. , 154. , 157.5, 157.5, 157.5, 157.5, 157.5,
                  157.5, 157.5, 157.5, 157.5, 157.5, 161. , 161. , 161. , 161. ,
                  161. , 161. , 161. , 161. , 161. , 161. , 164.5, 164.5, 164.5,
                  164.5, 164.5, 164.5, 164.5, 164.5, 164.5, 168. , 168. , 168. ,
                  168. , 168. , 168. , 168. , 168. , 171.5, 171.5, 171.5, 171.5,
                  171.5, 171.5, 171.5, 175. , 175. , 175. , 175. , 175. , 175. ,
                  178.5, 178.5, 178.5, 178.5, 182. , 182. , 185.5, 185.5, 189., 192.5])

print(f"Tamanho dos dados: {dados.size}")

print(f"Média: {dados.mean()}")
print(f"Desvio padrão: {dados.std(ddof=1)}")
print(f"Variância: {dados.var(ddof=1)}")
print(f"Mediana: {np.median(dados)}")
print(f"Máximo: {dados.max()}")
print(f"Mínimo: {dados.min()}")
print(f"Amplitude: {dados.max() - dados.min()}")

print("Gráfico de densidade com histograma")
sns.histplot(dados, kde=True)
plt.title("Distribuição de Alturas")
plt.xlabel("Altura (cm)")
plt.ylabel("Densidade")
plt.show()
