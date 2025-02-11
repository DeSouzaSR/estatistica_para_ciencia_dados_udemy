# Exercício
#O objetivo deste exercício é calcular o coeficiente e a taxa de evasão utilizando a tabela abaixo. 
#O índice é determinado pelo quantidade de alunas que se desmatricularam, dividio pelo total de matrículas. A taxa é o índice, multiplicado por 100.

import pandas as pd

ano_graudacao = [1, 2, 3, 4]
matriculas_marco = [70, 50, 47, 23]
matriculas_novembro = [65, 48, 40, 22]

df = pd.DataFrame(list(zip(ano_graudacao, matriculas_marco, matriculas_novembro)), columns=['Ano de Graduação', 'Matrículas em Março', 'Matrículas em Novembro'])

df['Desmatriculadas'] = df['Matrículas em Março'] - df['Matrículas em Novembro']

df['Índice de Evasão'] = df['Desmatriculadas'] / df['Matrículas em Março']  

df['Taxa de Evasão'] = df['Índice de Evasão'] * 100

print(df)