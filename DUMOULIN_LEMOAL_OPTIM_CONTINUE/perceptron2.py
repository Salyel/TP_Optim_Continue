# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
from mpmath import *
from random import *

class Perceptron2:
    
    #Constructor
    def __init__(self, input_size): 
        self.weights = np.random.random((input_size, 1)) 
        #self.weights = [1 for i in range(input_size)]
        #self.weights = [0.5 for i in range(input_size)]
        #self.weights = [0 for i in range(input_size)]
        
    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tangenteHyperbolique(self, x):
        return (1 / (1 + np.exp(-2*x))) + 1
    
    def miseAJourPoids(self, donnees, valeurs, lambdda, objectif):
        sommeDonnees = np.sum(donnees) 
        for i in range (len(self.weights)):
            if (objectif == 0):
                self.weights[i] = self.weights[i] - ((3 * lambdda * (sommeDonnees + 1) * valeurs * (1 - valeurs) * (valeurs - objectif)) * donnees[i])
            else:
                self.weights[i] = self.weights[i] - ((lambdda * (sommeDonnees + 1) * valeurs * (1 - valeurs) * (valeurs - objectif)) * donnees[i]) 
    
    def calculFonctionEntrainement(self, donnees, objectif, lambdda, amelioration):
        valeurs = []
        
        #Bout de code prenant toutes les données dans l'ordre
        for i in range (len(donnees)):
            valeurs.append(self.sigmoide((np.dot(donnees[i], self.weights))))
            if amelioration == True: 
                self.miseAJourPoids(donnees[i], valeurs[i], lambdda, objectif[i])
            
        #Bout de code prenant les données dans l'ordre suivant : 1er, 150e, 2e, 151e ...
        '''
        for i in range (int(len(donnees)/2)):
            valeurs.append(self.sigmoide((np.dot(donnees[i], self.weights))))
            valeurs.append(self.sigmoide((np.dot(donnees[i+int(len(donnees)/2)], self.weights))))
        ''' 
        
        return valeurs