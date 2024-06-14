"""
Estudos de Amostragem
"""
import pandas as pd
import sys
sys.path.insert(1, 'src')

from amostragem import amostra_aleatoria_simples, \
amostragem_sistematica, \
amostragem_estratificada, \
amostragem_agrupamento, \
amostragem_reservatorio

def main():
    dataset = pd.read_csv("data/raw_data/credit_data.csv")
    n_amostras = 1000

    df_amostra_aleatoria_simples = amostra_aleatoria_simples(dataset, n_amostras)
    df_amostra_sistematica = amostragem_sistematica(dataset, n_amostras)
    df_amostra_agrupamento = amostragem_agrupamento(dataset, n_amostras)
    df_amostra_estratificada = amostragem_estratificada(dataset, n_amostras / len(dataset))
    df_amostragem_reservatorio = amostragem_reservatorio(dataset, n_amostras)

    print(df_amostra_aleatoria_simples['age'].mean())
    print(df_amostra_sistematica['age'].mean())
    print(df_amostra_agrupamento['age'].mean())
    print(df_amostra_estratificada['age'].mean())
    print(df_amostragem_reservatorio['age'].mean())

if __name__ =="__main__":
    main()