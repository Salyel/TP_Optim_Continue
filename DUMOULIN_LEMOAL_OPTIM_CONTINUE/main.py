
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import perceptron
from perceptron import Perceptron

from sklearn import datasets
from sklearn.decomposition import PCA

'''
donnees = [[2, 3, 2, 2], [60, 3, 7, 2], [40, 3, 2, 8], [40, 9, 2, 2], [2, 2, 2, 2]]
objectif = [0, 1, 1, 1, 0]
lambdda = 0.1

donnees2 = [[2, 3, 2, 2], [60, 3, 7, 2], [20, 3, 2, 8], [40, 9, 2, 2], [20, 2, 2, 2], [2, 2, 2, 2], [40, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
objectif2 = [0, 1, 1, 1, 0, 1, 0, 1, 0]


for i in range (len(donnees)):
    donnees[i].append(1)

perceptron = Perceptron(4)
for i in range (500):
    print("// Itération", i)
    perceptron.calculFonctionEntrainement(donnees, objectif, lambdda)
''' 

def remplissageObjectif10Suivants(donnees_brutes):
    i = 1
    derniereValeurStep = 1
    objectif = []
    for k in range(1, 291):
        i = derniereValeurStep
        step = donnees_brutes['step'][i]
        augmentation = 0
        minimum = 9999999999
        while donnees_brutes['step'][i] == step:
            if (donnees_brutes['side'][i] == 'bids'):
                if donnees_brutes['price'][i] < minimum:
                    minimum = donnees_brutes['price'][i]
            i += 1
        derniereValeurStep = i
        while augmentation == 0 and donnees_brutes['step'][i] <= step+10:    #Tant qu'on sait pas encore si ça a augmenté, et qu'on est pas 10 steps plus loin
            if (donnees_brutes['side'][i] == 'asks') and (donnees_brutes['product_id'][i] == 'BTC-EUR'):     #Si c'est une demande et que c'est le même type d'échange
                if (donnees_brutes['price'][i] > minimum):   #Si la demande est plus élevée que le prix de l'offre
                    augmentation = 1
            objectif.append(augmentation)
            i += 1
    return objectif

donnees_brutes = pd.read_csv('ask_bid.csv', sep=";", index_col=0)
objectif = []
print(donnees_brutes)
#types = type(donnees_brutes['price'][5])
#print(types)
objectif = remplissageObjectif10Suivants(donnees_brutes)
print(objectif)

'''
Les données qu'on utilise ça va être blablabla
'''

