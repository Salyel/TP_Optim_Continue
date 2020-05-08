
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import perceptron
from perceptron import Perceptron
import traitementDeDonnees as traitement
from perceptronMomentum import PerceptronMomentum
from perceptronRMSProp import PerceptronRMS
import random
from sklearn import datasets
from sklearn.decomposition import PCA

#Récupération des données, puis assignation des objectifs et formatage
donnees_brutes = pd.read_csv('ask_bid.csv', sep=";")
objectif = []
objectif = traitement.remplissage10SuivantsProf(donnees_brutes)
donnees_formatees = []
donnees_formatees = traitement.formatageMin(donnees_brutes)  #On envoie uniquement le min des bids des 20 steps précédents, centré normé

for i in range (len(donnees_formatees)):
    donnees_formatees[i].append(1)

#Cette partie du fichier sert à définir les paramètres, le plus important étant le type de gradient à utiliser
lambdda = 0.7                             #Pas d'apprentissage (modifiable)                   
beta = 0.9                                #Paramètre pour le gradient avec momentum et le RMS (modifiable)
nb_epoch = 200                            #Le nombre d'itération pour notre apprentissage (modifiable)
nb_folds = 3                              #Le nombre de folds pour la validation croisée (modifiable)
input_size = len(donnees_formatees[0])    #La taille des données d'entrée, à ne pas modifier (non modifiable)
moyenne_pourcentage = 0                   #La moyenne des précisions sur tous les folds, à ne pas modifier (non modifiable)
typeGradient = "Momentum"                 #Peut prendre les valeurs "RMS", "Momentum" ou "", pour nos 3 perceptrons différents 

#On parcours tous les folds
for j in range (nb_folds):
    #On sélectionne le bon perceptron qui correspond à la descente de gradient souhaitée
    if (typeGradient == "RMS"):
        perceptron = PerceptronRMS(input_size)
    elif (typeGradient == "Momentum"):
        perceptron = PerceptronMomentum(input_size)
    else:
        perceptron = Perceptron(input_size)
      
    #Création des bases d'apprentissage et de validation, ainsi que l'ordre aléatoire dans lequel on traite les données
    base_apprentissage = traitement.separationBaseApprentissageFolds(donnees_formatees, j, nb_folds)
    base_validation = traitement.separationBaseValidationFolds(donnees_formatees, j, nb_folds)
    random_apprentissage = [i for i in range(len(base_apprentissage))]
    random_validation = [i for i in range(len(base_validation))]
    
    #A chaque itération de l'apprentissage
    for i in range (nb_epoch):
        random.shuffle(random_apprentissage)    #Application de l'ordre aléatoire
        
        #Calcul des valeurs d'activation de notre algorithme correspondant au gradient souhaité, et modification des poids
        valeur = []
        if (typeGradient == "RMS" or typeGradient == "Momentum"):
            valeur = perceptron.calculFonctionEntrainement(base_apprentissage, objectif, lambdda, True, random_apprentissage, beta)
        else:
            valeur = perceptron.calculFonctionEntrainement(base_apprentissage, objectif, lambdda, True, random_apprentissage)
    
    #Calcul des valeurs d'activation suite à l'apprentissage (False signifie qu'il ne faut plus modifier les poids)
    valeur = []
    if (typeGradient == "RMS" or typeGradient == "Momentum"):
        valeur = perceptron.calculFonctionEntrainement(base_validation, objectif, lambdda, False, random_validation, beta)
    else:
        valeur = perceptron.calculFonctionEntrainement(base_validation, objectif, lambdda, False, random_validation)

    #Calcul du pourcentage de précision
    pourcentage = traitement.precision(valeur, objectif, nb_folds, j)
    moyenne_pourcentage += pourcentage
    print("Le pourcentage de précision est de ", pourcentage)

moyenne_pourcentage /= nb_folds
print("La moyenne des pourcentages de précision est de ", moyenne_pourcentage)
