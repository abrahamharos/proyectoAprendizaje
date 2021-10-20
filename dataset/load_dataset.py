import numpy as np
import pandas as pd

# Case: String Example: '1.1'
# THIS SCRIPT SHOULD BE EXECUTED AT THE HOME OF THE REPO (Where the README.MD file is)
def load_dataset(case):
    XTrain = np.loadtxt('dataset/casos/' + case + '/XTrain.csv')
    XTest = np.loadtxt('dataset/casos/' + case + '/XTest.csv')
    yTrain = np.loadtxt('dataset/casos/' + case + '/yTrain.csv')
    yTest = np.loadtxt('dataset/casos/' + case + '/yTest.csv')
    
    return XTrain, XTest, yTrain, yTest

if __name__ == "__main__":
    load_dataset('1.1')
