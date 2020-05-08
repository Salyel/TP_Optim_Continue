# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
from mpmath import *
from random import *

#Ce perceptron implémente l'algorithme de descente de gradient avec Momentum.
class PerceptronMomentum:
    
    #Les poids sont initialisés aléatoirement entre 0 et 1 lorsque le perceptron est créé
    #La moyenne exponentielle est initialisée à 0.
    def __init__(self, input_size): 
        self.weights = np.random.random((input_size, 1)) 
        self.exponential_mean = [0 for i in range(input_size)]
        
    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tangenteHyperbolique(self, x):
        return (1 / (1 + np.exp(-2*x))) + 1
    
    #Fonction qui met à jour les poids.
    def miseAJourPoids(self, donnees, valeurs, lambdda, objectif, beta):
        sommeDonnees = np.sum(donnees) 
        #print("La somme de la donnée est égale à", sommeDonnees)
        #print("valeur égale à", valeurs)
        #print("l'objectif est égale à", objectif)
        for i in range (len(self.weights)):
            grad = (sommeDonnees + 1) * valeurs * (1 - valeurs) * (valeurs - objectif) * donnees[i]
            if self.exponential_mean[i] == 0:
                self.exponential_mean[i] = grad
            else:
                self.exponential_mean[i] = beta*self.exponential_mean[i] + (1 - beta)*grad

            if (objectif == 0):     #Lorsque objectif == 0, on met un facteur 1.5 à la modification puisqu'on essaie d'apprendre plus des rares cas.
                self.weights[i] = self.weights[i] - 1.5*lambdda*self.exponential_mean[i]
            else:
                self.weights[i] = self.weights[i] - lambdda*self.exponential_mean[i]
      
    #Fonction à appeler pour lancer l'apprentissage des données si amelioration == True, ou juste pour obtenir des valeurs avec amelioration == False
    def calculFonctionEntrainement(self, donnees, objectif, lambdda, amelioration, random_order, beta):
        valeurs = []

        for i in range (len(donnees)):
            valeurs.append(self.sigmoide((np.dot(donnees[random_order[i]], self.weights))))
            if amelioration == True: 
                self.miseAJourPoids(donnees[random_order[i]], valeurs[i], lambdda, objectif[random_order[i]], beta)

        return valeurs

