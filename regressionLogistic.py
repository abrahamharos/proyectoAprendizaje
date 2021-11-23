import load_dataset as ld
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Casos en los que se usará regresion logistica:
#   Caso 3.1: Km/l Clasificar instancia por calificacion de contaminacion del aire.
#   Caso 4.1: Km/l Clasificar instancia por calificacion de emisiones de gas.

cases = {"3.1","4.1"}

print(" ~ Regresion Logistic ~ ")

for case in cases:

    XTrain, XTest, yTrain, yTest = ld.load_dataset(case)

    # Create and train the multiple regression model
    LR = LogisticRegression(max_iter=2500)
    LR.fit(XTrain, yTrain)
    prediction = LR.predict(XTest)

    # Test the multiple regression model
    accuracy = LR.score(XTest, yTest)

    meanSquareError = mean_squared_error(yTest, prediction)
    print("\n\nCase: " + case)
    print("Accuracy:", accuracy)
    print("Mean Squared Error:", meanSquareError)

    np.savetxt('roc/' + str(case) + '/logistica/yTest.csv', yTest)
    np.savetxt('roc/' + str(case) + '/logistica/yPredicted.csv', prediction)
