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
    dataset = pd.read_csv('data/raw_data/census.csv')

    df_amostra_aleatoria_simples = amostra_aleatoria_simples(dataset, 100)
    df_amostra_sistematica = amostragem_sistematica(dataset, 100)
    df_amostra_agrupamento = amostragem_agrupamento(dataset, 325)

    # O valor 0.0030711587481956942 refere-se a 100 / len(dataset), ou seja,
    # o tamanho relativo da amostra
    df_amostra_estratificada = amostragem_estratificada(dataset, 0.0030711587481956942)
    
    df_amostragem_reservatorio = amostragem_reservatorio(dataset, 100)


    # Write out
    print("df_amostra_aleatoria_simples\n")
    print(amostra_aleatoria_simples(dataset, 100))

    print("df_amostra_sistematica\n")
    print(amostragem_sistematica(dataset, 100))

    print("df_amostra_agrupamento\n")
    print(amostragem_agrupamento(dataset, 325))

    print("df_amostra_estratificada\n")
    print(amostragem_estratificada(dataset, 0.0030711587481956942))

    print("df_amostra_reservatorio\n")
    print(amostragem_reservatorio(dataset, 100))    


if __name__ == '__main__':
    main()