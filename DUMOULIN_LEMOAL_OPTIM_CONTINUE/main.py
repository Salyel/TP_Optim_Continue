
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import perceptron
from perceptron import Perceptron

from sklearn import datasets
from sklearn.decomposition import PCA


#donnees_brutes = pd.read_csv('data_buy_sell.csv', sep=";", index_col=0)

#print(donnees_brutes)

#print(donnees_brutes["time"][4])
        
#perceptron = Perceptron(donnees_brutes)
#perceptron.imprimer()

donnees = [[2, 3, 2, 2], [60, 3, 7, 2], [40, 3, 2, 8], [40, 9, 2, 2], [2, 2, 2, 2]]
objectif = [0, 1, 1, 1, 0]
lambdda = 0.1

donnees2 = [[2, 3, 2, 2], [60, 3, 7, 2], [20, 3, 2, 8], [40, 9, 2, 2], [20, 2, 2, 2], [2, 2, 2, 2], [40, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
objectif2 = [0, 1, 1, 1, 0, 1, 0, 1, 0]

iris = datasets.load_iris()
X = iris.data[:100, :2]  # we only take the first two features.
y = iris.target[:100]
#print(X)
#print(y)

Xo = []
for i in range (len(X)):
    Xoo = []
    for j in range (len(X[i])):
        Xoo.append(X[i][j])
    Xoo.append(1)
    Xo.append(Xoo)

for i in range (len(donnees)):
    donnees[i].append(1)

perceptron = Perceptron(4)
for i in range (500):
    print("// Itération", i)
    perceptron.calculFonctionEntrainement(donnees, objectif, lambdda)


