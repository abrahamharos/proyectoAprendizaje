import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print(" ~ Reading dataframe_proyecto.csv")
data = pd.read_csv('dataframe_proyecto.csv')

# Aqui se añaden nuevos casos de uso para el dataset

# Caso 1.1: Predecir Km/l en carretera dados cilindros, potencia y modelo.
X1 = data.iloc[:, 3].values.reshape(-1, 1) # Modelo
X2 = data.iloc[:, 6:8].values.reshape(-1, 2) # Cilindros y potencias
X = np.append(X1, X2, 1) # [MODELO, CILINDROS, AÑO]

y = data.iloc[:, 11].values.reshape(-1, 1) # Km/l en carretera

# Split dataset 80% train 20% test
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

np.savetxt('../casos/1.1/XTrain.csv', XTrain)
np.savetxt('../casos/1.1/XTest.csv', XTest)
np.savetxt('../casos/1.1/yTrain.csv', yTrain)
np.savetxt('../casos/1.1/yTest.csv', yTest)
print(" ~ Caso 1.1 Guardado en ../casos/1.1/")


# Caso 1.2: Predecir Km/l en ciudad dados cilindros, potencia y modelo.
X1 = data.iloc[:, 3].values.reshape(-1, 1) # Modelo
X2 = data.iloc[:, 6:8].values.reshape(-1, 2) # Cilindros y potencias
X = np.append(X1, X2, 1) # [MODELO, CILINDROS, AÑO]

y = data.iloc[:, 10].values.reshape(-1, 1) # Km/l en ciudad

# Split dataset 80% train 20% test
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

np.savetxt('../casos/1.2/XTrain.csv', XTrain)
np.savetxt('../casos/1.2/XTest.csv', XTest)
np.savetxt('../casos/1.2/yTrain.csv', yTrain)
np.savetxt('../casos/1.2/yTest.csv', yTest)
print(" ~ Caso 1.2 Guardado en ../casos/1.2/")


# Caso 1.3: Predecir Km/l en combinados (carretera y ciudad) dados cilindros, potencia y modelo.
X1 = data.iloc[:, 3].values.reshape(-1, 1) # Modelo
X2 = data.iloc[:, 6:8].values.reshape(-1, 2) # Cilindros y potencias
X = np.append(X1, X2, 1) # [MODELO, CILINDROS, AÑO]

y = data.iloc[:, 12].values.reshape(-1, 1) # Km/l Combinados (carretera y ciudad)

# Split dataset 80% train 20% test
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

np.savetxt('../casos/1.3/XTrain.csv', XTrain)
np.savetxt('../casos/1.3/XTest.csv', XTest)
np.savetxt('../casos/1.3/yTrain.csv', yTrain)
np.savetxt('../casos/1.3/yTest.csv', yTest)
print(" ~ Caso 1.3 Guardado en ../casos/1.3/")

# Caso 2.1: Predecir CO2 dados km/l en carretera, potencia, modelo y cilindros.

# Caso 2.2: Predecir CO2 dados km/l en ciudad, potencia, modelo y cilindros.

# Caso 2.3: Predecir CO2 dados km/l combinados (ciudad y carretera), potencia, modelo y cilindros.

# Caso 3.1: Clasificar la instancia por calificación de contaminación del aire

# Caso 4.1: Clasificar la instancia por calificación de emisiones de gas

# Caso 5.1: Utilizar el tipo de transmisión (binario) como clasificador usando caballos de fuerza y otros atributos numéricos como variables independientes.
