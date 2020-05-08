# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
from mpmath import *
from random import *

#Ce perceptron est le premier que nous avons développé sur les séances de TP.
#Il implémente l'algorithme de descente de gradient steepest descent.
class Perceptron:
    
    #Les poids sont initialisés aléatoirement entre 0 et 1 lorsque le perceptron est créé
    def __init__(self, input_size): 
        self.weights = np.random.random((input_size, 1)) 
        
    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tangenteHyperbolique(self, x):
        return (1 / (1 + np.exp(-2*x))) + 1
    
    #Fonction qui met à jour les poids.
    def miseAJourPoidsStochastique(self, donnees, valeurs, lambdda, objectif):
        sommeDonnees = np.sum(donnees) 
        #print("La somme de la donnée est égale à", sommeDonnees)
        #print("valeur égale à", valeurs)
        #print("l'objectif est égale à", objectif)
        for i in range (len(self.weights)):
            if (objectif == 0):     #Lorsque objectif == 0, on met un facteur 1.5 à la modification puisqu'on essaie d'apprendre plus des rares cas.
                self.weights[i] = self.weights[i] - (1.5 * lambdda * (sommeDonnees + 1) * valeurs * (1 - valeurs) * (valeurs - objectif) * donnees[i])
            else:
                self.weights[i] = self.weights[i] - (1 * lambdda * (sommeDonnees + 1) * valeurs * (1 - valeurs) * (valeurs - objectif) * donnees[i])
        
    #Fonction à appeler pour lancer l'apprentissage des données si amelioration == True, ou juste pour obtenir des valeurs avec amelioration == False
    def calculFonctionEntrainement(self, donnees, objectif, lambdda, amelioration, random_order):
        valeurs = []

        for i in range (len(donnees)):
            valeurs.append(self.sigmoide((np.dot(donnees[random_order[i]], self.weights))))
            if amelioration == True: 
                self.miseAJourPoidsStochastique(donnees[random_order[i]], valeurs[i], lambdda, objectif[random_order[i]])

        return valeurs