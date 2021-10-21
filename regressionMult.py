import load_dataset as ld
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Casos en los que se usará arboles de decisión:
#   Caso 1.1: Km/l en carretera dados cilindros, potencia y modelo.
#   Caso 1.2: Km/l en ciudad dados cilindros, potencia y modelo.
#   Caso 1.3: Km/l (en carretera y en ciudad) dados cilindros, potencia y modelo.

#   Caso 2.1: C02 dados Km/l en carretera, potencia, modelo y cilindros.
#   Caso 2.2: C02 dados Km/l en ciudad, potencia, modelo y cilindros.
#   Caso 2.3: C02 dados Km/l (en carretera y ciudad), potencia, modelo y cilindros.

cases = {"1.1","1.2","1.3","2.1","2.2","2.3"}

print(" ~ Regresion multi-variable ~ ")

for case in cases:

    XTrain, XTest, yTrain, yTest = ld.load_dataset(case)

    # Create and train the multiple regression model
    LR = LinearRegression()
    LR.fit(XTrain, yTrain)
    prediction = LR.predict(XTest)

    # Test the multiple regression model
    accuracy = LR.score(XTest, yTest)

    meanSquareError = mean_squared_error(yTest, prediction)
    print("\n\nCase: " + case)
    print("Accuracy:", accuracy)
    print("Mean Squared Error:", meanSquareError)
