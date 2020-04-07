# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
from random import *

class Perceptron:
    
    #Constructor
    def __init__(self, donnees, objectif):
        self.donnees = donnees
        self.weights = np.random.random((4,1))
        self.bias = random()*5
        print(self.bias)
        self.cout = 0
        self.valeurs = np.zeros((len(donnees), 1))
        self.objectif = objectif
        
    def sigmoide(self, x):
        return 1 / (1 + math.exp(-x))
    
    def tangenteHyperbolique(self, x):
        return (1 / (1 + math.exp(-2*x))) + 1
        
    def calculFonctionActivation(self):
        for i in range (len(self.donnees)):
            self.valeurs[i] = self.sigmoide((np.dot(self.donnees[i], self.weights) + self.bias))
        print(self.valeurs)
        for i in range (len(self.valeurs)):
            self.cout += (self.valeurs[i] - self.objectif)**2
        print(self.cout)
        if (self.cout > 4):
            print('méchant algorithme nul')
        else:
            print('c bien')
            
    