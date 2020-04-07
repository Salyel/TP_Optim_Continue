
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import perceptron
from perceptron import Perceptron


#donnees_brutes = pd.read_csv('data_buy_sell.csv', sep=";", index_col=0)

#print(donnees_brutes)

#print(donnees_brutes["time"][4])
        
#perceptron = Perceptron(donnees_brutes)
#perceptron.imprimer()

donnees = np.random.random((10,4))
objectif = 0.3
#print(donnees)

perceptron = Perceptron(donnees, objectif)
perceptron.calculFonctionActivation()