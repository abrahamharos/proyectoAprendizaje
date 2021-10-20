import random
import pandas as pd
import numpy as np
import statistics

#lectura de datos
data_text = pd.read_excel('Consumo Gasolina Autos Ene 2018.xlsx')
print(data_text)

data_text.loc[data_text['Calificación Contam. Aire']=='?', 'Calificación Contam. Aire'] = np.NaN
x = data_text['Calificación Contam. Aire'].dropna()
media = np.asarray(x, dtype=np.float).mean()
s = statistics.pstdev(np.asarray(x, dtype=np.float))
data_text = pd.read_excel('Consumo Gasolina Autos Ene 2018.xlsx')
data_text.loc[data_text['Calificación Contam. Aire']=='?', 'Calificación Contam. Aire'] = random.randint(int(media-s), int(media+s))
data_text['Calificación Contam. Aire']

# TODO: Make this a relative path
data_text.to_csv (r'C:\\Users\\Andres\\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\\Semestre 9\\Aprendizaje Automático\\dataframe_proyecto.csv', index = False, header=True)