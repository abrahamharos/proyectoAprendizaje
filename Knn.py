import load_dataset as ld
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

XTrain, XTest, yTrain, yTest = ld.load_dataset('3.1')

n_neighbors = [1, 2, 3, 5, 10, 15, 20, 50, 75, 100]
values = []
maxModelScore, maxModelIndex = 0, 0
maxModel = {}

for i in range(len(n_neighbors)):
    modelTemp = KNeighborsClassifier(n_neighbors=n_neighbors[i], algorithm='brute')
    modelTemp.fit(XTrain, yTrain)
    Y_predTemp = modelTemp.predict(XTest)
    values.append(float(modelTemp.score(XTest, yTest)))
    print("N-Neighbors: {} - Score: {}".format(n_neighbors[i],values[-1]))
    if values[-1] > maxModelScore:
        maxModelScore = values[-1]
        maxModelIndex = i
        maxModel = Y_predTemp

print("\nCase 3.1")
print("N-Neighbors: {}".format(n_neighbors[maxModelIndex]))
print("Accuracy: {}\n".format(maxModelScore))
np.savetxt('roc/3.1/knn/yTest.csv', yTest)
np.savetxt('roc/3.1/knn/yPredicted.csv', Y_predTemp)


XTrain, XTest, yTrain, yTest = ld.load_dataset('4.1')

n_neighbors = [1, 2, 3, 5, 10, 15, 20, 50, 75, 100]
values = []
maxModelScore, maxModelIndex = 0, 0
maxModel = {}

for i in range(len(n_neighbors)):
    modelTemp = KNeighborsClassifier(n_neighbors=n_neighbors[i], algorithm='brute')
    modelTemp.fit(XTrain, yTrain)
    Y_predTemp = modelTemp.predict(XTest)
    values.append(float(modelTemp.score(XTest, yTest)))
    print("N-Neighbors: {} - Score: {}".format(n_neighbors[i],values[-1]))
    if values[-1] > maxModelScore:
        maxModelScore = values[-1]
        maxModelIndex = i
        maxModel = Y_predTemp

print("\nCase 4.1")
print("N-Neighbors: {}".format(n_neighbors[maxModelIndex]))
print("Accuracy: {}".format(maxModelScore))
np.savetxt('roc/4.1/knn/yTest.csv', yTest)
np.savetxt('roc/4.1/knn/yPredicted.csv', Y_predTemp)