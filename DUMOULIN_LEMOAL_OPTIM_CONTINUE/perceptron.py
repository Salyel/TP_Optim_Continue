# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
from mpmath import *
from random import *

class Perceptron:
    
    #Constructor
    def __init__(self, input_size): 
        #self.weights = np.random.random((input_size, 1)) 
        #self.weights = [1 for i in range(input_size)]
        self.weights = [0.5 for i in range(input_size)]
        #self.weights = [0 for i in range(input_size)]
        
    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tangenteHyperbolique(self, x):
        return (1 / (1 + np.exp(-2*x))) + 1
    
    def miseAJourPoids(self, donnees, valeurs, lambdda, objectif):
        for j in range (len(donnees)):
            #print("Début de l'entrainement sur la donnée", j)
            sommeDonnees = np.sum(donnees[j]) 
            #print("La somme de la donnée est égale à", sommeDonnees)
            #print("valeur égale à", valeurs[j])
            #print("l'objectif est égale à", objectif[j])
            for i in range (len(self.weights)):
                if (objectif[j] == 0):
                    self.weights[i] = self.weights[i] - ((3 * lambdda * (sommeDonnees + 1) * valeurs[j] * (1 - valeurs[j]) * (valeurs[j] - objectif[j])) * donnees[j][i])
                else:
                    self.weights[i] = self.weights[i] - ((lambdda * (sommeDonnees + 1) * valeurs[j] * (1 - valeurs[j]) * (valeurs[j] - objectif[j])) * donnees[j][i]) 
        
    def calculFonctionEntrainement(self, donnees, objectif, lambdda, amelioration):
        valeurs = []
        
        #Bout de code prenant toutes les données dans l'ordre
        for i in range (len(donnees)):
            valeurs.append(self.sigmoide((np.dot(donnees[i], self.weights))))
            
        #Bout de code prenant les données dans l'ordre suivant : 1er, 150e, 2e, 151e ...
        '''
        for i in range (int(len(donnees)/2)):
            valeurs.append(self.sigmoide((np.dot(donnees[i], self.weights))))
            valeurs.append(self.sigmoide((np.dot(donnees[i+int(len(donnees)/2)], self.weights))))
        ''' 
        if amelioration == True: 
            self.miseAJourPoids(donnees, valeurs, lambdda, objectif)
        
        return valeurs