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
