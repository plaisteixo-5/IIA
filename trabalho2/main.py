import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score, classification_report
from sklearn.tree import export_graphviz
import pydot
from sklearn.inspection import permutation_importance

data = pd.read_csv("diabetes.csv")
data.head()


def calculateForest(estimators, train_testers):

    # Aplica o método train_test_split para gerar randomicamente os testes e treinos que serão utilizados
    # para modelar as florestas geradas.
    xTrain, xTest, yTrain, yTest = train_testers

    # Treino da floresta utilizando o número de estimadores passados no cabeçalho da função.
    randomForest = RandomForestClassifier(
        n_estimators=estimators, random_state=0)

    # Ajusta o modelo da floresta de acordo com os treinos produzidos.
    randomForest.fit(xTrain, yTrain)

    # Prevê o resultado baseado no modelo ajustado anteriormente.
    predictValue = randomForest.predict(xTest)

    print(f"="*80)
    print(confusion_matrix(yTest, predictValue))
    print(classification_report(yTest, predictValue))
    print(accuracy_score(yTest, predictValue))

    random_forest_small = RandomForestClassifier(
        n_estimators=estimators, max_depth=3)
    random_forest_small.fit(xTrain, yTrain)
    estimators_size = len(random_forest_small.estimators_)
    tree_small = random_forest_small.estimators_[estimators_size-1]

    export_graphviz(tree_small,
                    out_file=f'small_tree{estimators}.dot',
                    feature_names=list(x.columns),
                    rounded=True,
                    precision=1)

    (graph, ) = pydot.graph_from_dot_file(f'small_tree{estimators}.dot')
    graph.write_png(f'small_tree{estimators}.png')


x = data.drop("Outcome", axis=1)
y = data["Outcome"]

xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, train_size=0.8, random_state=0)

calculateForest(10, [xTrain, xTest, yTrain, yTest])
calculateForest(100, [xTrain, xTest, yTrain, yTest])
calculateForest(int(np.sqrt(len(x.columns))), [xTrain, xTest, yTrain, yTest])
