
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import perceptron
from perceptron import Perceptron
from perceptron2 import Perceptron2
import traitementDeDonnees as traitement
import random

from sklearn import datasets
from sklearn.decomposition import PCA

donnees_brutes = pd.read_csv('ask_bid.csv', sep=";")
objectif = []
objectif = traitement.remplissage10SuivantsProf(donnees_brutes)
donnees_formatees = []
donnees_formatees = traitement.formatageVariationMaxMin(donnees_brutes)
#print(donnees_formatees)
lambdda = 0.1

for i in range (len(donnees_formatees)):
    donnees_formatees[i].append(1)
    
nb_epoch = 100
nb_folds = 5
moyenne_pourcentage = 0
input_size = len(donnees_formatees[0])
for j in range (nb_folds):
    perceptron = Perceptron(input_size)
    base_apprentissage = traitement.separationBaseApprentissageFolds(donnees_formatees, j, nb_folds)
    base_validation = traitement.separationBaseValidationFolds(donnees_formatees, j, nb_folds)
    for i in range (nb_epoch):
       # print("// Itération", i)
        random.shuffle(base_apprentissage)
        valeur = []
        valeur = perceptron.calculFonctionEntrainement(base_apprentissage, objectif, lambdda, True)
    random.shuffle(base_validation)
    valeur = []
    valeur = perceptron.calculFonctionEntrainement(base_validation, objectif, lambdda, False)
    pourcentage = traitement.accuracy(valeur, objectif)
    moyenne_pourcentage += pourcentage
    print("Le pourcentage de précision est de ", pourcentage)
moyenne_pourcentage /= nb_folds
print("La moyenne des pourcentages de précision est de ", moyenne_pourcentage)



'''
Pour la prochaine fois : Faire en sorte de répartir des données d'apprentissage pour que ce soit pas toujours
dans le même ordre, ou bien faire base apprentissage / base de test / base de validation.
'''

'''
Les données qu'on utilise ça va être le price:
    - couper en pleins de morceaux (en 3 par 3 de step), et on regarde la diff entre les évolutions des différences
    des moyennes
    
    - (moyenne des différences à la moyenne de notre step - moyenne des différences à la moyenne de chaque autre step)
'''

