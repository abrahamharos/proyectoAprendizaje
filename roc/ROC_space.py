import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

def calculateROC_Multiclass(testy, predicty, classifier_name, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(testy[:, i], predicty[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print(fpr['macro'])
    print(tpr['macro'])
    return [classifier_name, fpr["macro"], tpr["macro"], roc_auc["macro"]]


# Plot all ROC curves
def plotROCs(ROCs, name):
    lw = 2
    plt.figure(figsize=(12, 7), dpi=100)

    colors = cycle(['black', 'darkorange', 'cornflowerblue', 'olive', 'gray', 'rosybrown', 'orange', 'darkviolet', 'crimson', 'slategray'])
    for i, color in zip(range(len(ROCs)), colors):
        plt.plot(ROCs[i][1], ROCs[i][2], color=color, lw=lw,
            label=ROCs[i][0] + ' (area = {0:0.2f})'
                    ''.format(ROCs[i][3]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Espacio ROC comparando los diferentes clasificadores para el caso ' + str(name))
    plt.legend(loc="lower right")
    plt.savefig(str(name) + '_ROC_curve.png')
    plt.show()

# List of cases and its corresponding models
cases = {
    "4.1": ['decision_tree', 'knn', 'logistica']
}

for case in cases:
    ROCs = []
    for model in cases[case]:
        classifier_name = model
        
        yTest = np.loadtxt('roc/' + str(case) + '/' + model + '/yTest.csv')
        yTest = label_binarize(yTest, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        yPredicted = np.loadtxt('roc/' + str(case) + '/' + model + '/yPredicted.csv')
        yPredicted = label_binarize(yPredicted, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        n_classes = yTest.shape[1]
        
        ROCs.append(calculateROC_Multiclass(yTest, yPredicted, classifier_name, n_classes))

    plotROCs(ROCs, case)

print(len(ROCs))
ROCs[0] = ['Decision Tree', np.array([0.        , 0.003003  , 0.00302115, 0.0030303 , 0.00307692,
       0.00308642, 0.00311526, 0.003125  , 0.0031348 , 0.0060423 ,
       0.00615385, 0.00623053, 0.00626959, 0.00906344, 0.00923077,
       0.00934579, 0.009375  , 0.00940439, 0.01230769, 0.01246106,
       0.0125    , 0.01253918, 0.01538462, 0.01557632, 0.01869159,
       0.01875   , 0.01880878, 0.02461538, 0.02492212, 0.028125  ,
       0.03134796, 0.03164557, 0.04361371, 0.04923077, 0.05642633,
       0.06329114, 0.07165109, 0.07788162, 0.10344828, 0.11782477,
       0.22429907, 0.26168224, 0.28037383, 1.        ]), np.array([0.7846258 , 0.7883295 , 0.79867433, 0.81534099, 0.83248385,
       0.83526163, 0.83782573, 0.84782573, 0.85026476, 0.86060958,
       0.86632387, 0.87914438, 0.88646146, 0.89335801, 0.89621515,
       0.90647156, 0.90897156, 0.91384961, 0.91670675, 0.91927085,
       0.92177085, 0.92420988, 0.92992416, 0.93505237, 0.94530878,
       0.94780878, 0.9502478 , 0.95881923, 0.96138334, 0.96388334,
       0.96632236, 0.96859509, 0.97115919, 0.97401633, 0.97645536,
       0.97872808, 0.98385629, 0.98642039, 0.98885942, 0.99230769,
       0.99487179, 0.9974359 , 1.        , 1.        ]), 0.9949512131598636]

ROCs[1] = ['Knn', np.array([0.        , 0.003003  , 0.00625   , 0.01898734, 0.03636364,
       0.04012346, 0.04361371, 0.04984424, 0.05538462, 0.06583072,
       0.09365559, 1.        ]), np.array([0.        , 0.18471896, 0.27632133, 0.49338604, 0.70435122,
       0.74001726, 0.76568602, 0.79760522, 0.81877288, 0.84267481,
       0.87179902, 1.        ]), 0.9118022704299431]

ROCs[2] = ['Regresion logistica', np.array([0.        , 0.00311526, 0.0031348 , 0.00626959, 0.00940439,
       0.01869159, 0.03115265, 0.04      , 0.125     , 1.        ]), np.array([0.96987738, 0.97244148, 0.9846366 , 0.98707563, 0.98951465,
       0.99207875, 0.99464286, 0.9975    , 1.        , 1.        ]), 0.9995194925729919]
plotROCs(ROCs, '3.1')
