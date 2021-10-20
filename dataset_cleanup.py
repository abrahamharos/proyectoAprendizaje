from pydataset import data
import random
import pandas as pd
import math
import numpy as np
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model
import seaborn as sb
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

matplotlib notebook

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

data_text.to_csv (r'C:\\Users\\Andres\\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\\Semestre 9\\Aprendizaje Automático\\dataframe_proyecto.csv', index = False, header=True)