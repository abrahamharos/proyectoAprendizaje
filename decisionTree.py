import os
import load_dataset as ld
import numpy as np
from graphviz import Source
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import mean_squared_error

# Casos en los que se usará arboles de decisión:
#   Caso 3.1: Clasificar la instancia por calificación de contaminación del aire.
#   Caso 4.1: Clasificar la instancia por calificación de eficiencia en emisiones de gas.
cases = {
    "3.1": {
        "feature_names": ['Modelo', 'Cilindros', 'Potencia', 'Tamaño', 'R. Combinado', 'CO2', 'NOx'],
        "target_names": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        "tree_filename": "Caso 3-1"
    },
    "4.1": {
        "feature_names": ['Modelo', 'Cilindros', 'Potencia', 'Tamaño', 'R. Combinado', 'CO2', 'NOx'],
        "target_names": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        "tree_filename": "Caso 4-1"
    }
}

# Setup folders
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

print(" ~ Arboles de decisión ~ ")

for case in cases:
    feature_names = cases[case]['feature_names']
    target_names = cases[case]['target_names']
    
    XTrain, XTest, yTrain, yTest = ld.load_dataset(case)

    # Create and train the decision tree model
    tree_clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=8)
    tree_clf.fit(XTrain, yTrain)

    predictedValues = tree_clf.predict(XTest)

    # Test the decision tree
    accuracy = tree_clf.score(XTest, yTest)

    meanSquareError = mean_squared_error(yTest, predictedValues)
    print("\n\nCase: " + case)
    print("Accuracy:", accuracy)
    print("Mean Squared Error:", meanSquareError)

    # Export an image of the tree
    dot_src = export_graphviz(tree_clf, feature_names=feature_names,
                            class_names=target_names, rounded=True,
                            filled=True)
    
    treeFilename = cases[case]['tree_filename']
    image_filename = os.path.join(IMAGES_PATH, treeFilename)
    Source(dot_src).render(image_filename, format="png", cleanup=True)

    np.savetxt('roc/' + str(case) + '/decision_tree/yTest.csv', yTest)
    np.savetxt('roc/' + str(case) + '/decision_tree/yPredicted.csv', predictedValues)
