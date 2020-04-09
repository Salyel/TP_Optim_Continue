# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
from random import *

class Perceptron:
    
    #Constructor
    def __init__(self, input_size): 
        self.weights = np.random.random((input_size + 1, 1))
       # self.weights = [1.2, 1, 1, 1, 1]
        
    def sigmoide(self, x):
        return 1 / (1 + math.exp(-x))
    
    def tangenteHyperbolique(self, x):
        return (1 / (1 + math.exp(-2*x))) + 1
    
    def miseAJourPoids(self, donnees, valeurs, lambdda, objectif):
        for j in range (len(donnees)):
            print("Début de l'entrainement sur la donnée", j)
            sommeDonnees = np.sum(donnees[j]) 
            print("La somme de la donnée est égale à", sommeDonnees)
            print("valeur égale à", valeurs[j])
            for i in range (len(self.weights)):
                self.weights[i] = self.weights[i] - ((lambdda * (sommeDonnees + 1) * valeurs[j] * (1 - valeurs[j]) * (valeurs[j] - objectif[j])) * donnees[j][i]) 
        
    def calculFonctionEntrainement(self, donnees, objectif, lambdda):
        valeurs = []
        for i in range (len(donnees)):
            #donnees[i].append(1)
            valeurs.append(self.sigmoide((np.dot(donnees[i], self.weights))))
        for i in range (len(self.weights)):
            print(self.weights[i]) 
        self.miseAJourPoids(donnees, valeurs, lambdda, objectif)

                
            
            
                