import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import perceptron
from perceptron import Perceptron
from traitementDeDonnees import *
import random
import statistics

from sklearn import datasets
from sklearn.decomposition import PCA

#Fonction formattant les données que l'on va donner au Perceptron.
#Renvoie les différences entre les moyennes bids et les moyennes asks de chaque step et des 20 précédents. (centré normé)
def formatage(commandes):
    donnees_formatees = []
    commandes = commandes[commandes["product_id"] =="BTC-EUR"]
    for step in range(21, 291):
        donnees_par_step = []
        instep = commandes[commandes["step"] == step]
        ask = instep[instep["side"] == "asks"]
        bid = instep[instep["side"] == "bids"]
        meanbid_actuel = bid["price"].mean()
        meanask_actuel = ask["price"].mean()
        
        for step_davant in range(step-20, step):
            instep_davant = commandes[commandes["step"] == step_davant]
            ask_davant = instep_davant[instep_davant["side"] == "asks"]
            bid_davant = instep_davant[instep_davant["side"] == "bids"]
            meanbid_davant = bid_davant["price"].mean()
            meanask_davant = ask_davant["price"].mean()
            
            donnees_par_step.append(meanbid_actuel-meanbid_davant)
            donnees_par_step.append(meanask_actuel-meanask_davant)
         
        #Centrage et normage des données
        moyenne = statistics.mean(donnees_par_step)
        ecarttype = statistics.stdev(donnees_par_step)
        for i in range (len(donnees_par_step)):
            donnees_par_step[i] = (donnees_par_step[i] - moyenne)/ecarttype   
         
        donnees_formatees.append(donnees_par_step)
            
    return donnees_formatees

#Fonction formattant les données que l'on va donner au Perceptron.
#Renvoie le min des bids des 20 précédents steps à chaque step. (centré normé)
def formatageMin(commandes):
    donnees_formatees = []
    commandes = commandes[commandes["product_id"] =="BTC-EUR"]
    for step in range(21, 291):
        donnees_par_step = []
        
        for step_davant in range(step-20, step):
            instep_davant = commandes[commandes["step"] == step_davant]
            bid_davant = instep_davant[instep_davant["side"] == "bids"]
            minimum = bid_davant["price"].min()
            
            donnees_par_step.append(minimum)
          
        #Centrage et normage des données
        moyenne = statistics.mean(donnees_par_step)
        ecarttype = statistics.stdev(donnees_par_step)
        for i in range (len(donnees_par_step)):
            donnees_par_step[i] = (donnees_par_step[i] - moyenne)/ecarttype
            
        donnees_formatees.append(donnees_par_step)
        
    
    return donnees_formatees

#Fonction formattant les données que l'on va donner au Perceptron.
#Renvoie la différence entre le maxbid du step actuel et le minask des 20 précédents pour chaque step. (centré normé)
def formatageVariationMaxMin(commandes):
    donnees_formatees = []
    commandes = commandes[commandes["product_id"] =="BTC-EUR"]
    for step in range(21, 291):
        donnees_par_step = []
        instep = commandes[commandes["step"] == step]
        bid = instep[instep["side"] == "bids"]
        maxbid = bid["price"].max()
        
        for step_davant in range(step-20, step):
            instep_davant = commandes[commandes["step"] == step_davant]
            ask_davant = instep_davant[instep_davant["side"] == "asks"]
            minask_davant = ask_davant["price"].min()
            donnees_par_step.append((maxbid-minask_davant))
         
        #Centrage et normage des données
        moyenne = statistics.mean(donnees_par_step)
        ecarttype = statistics.stdev(donnees_par_step)
        for i in range (len(donnees_par_step)):
            donnees_par_step[i] = (donnees_par_step[i] - moyenne)/ecarttype   
            
        donnees_formatees.append(donnees_par_step)
            
    return donnees_formatees


#Fonction remplissant la liste des objectifs à atteindre pour chaque donnée.
#Si le maxbid du step actuel est supérieur minask des 20 step suivants, alors objectif = 1, 0 sinon.
def remplissage10SuivantsProf(commandes):
    objectif = []
    compteur0 = 0
    compteur1 = 0
    commandes = commandes[commandes["product_id"] =="BTC-EUR"]
    for step in range(1,291):
        instep = commandes[commandes["step"] == step]
        tenstep = commandes[commandes["step"] > step]
        tenstep = tenstep[tenstep["step"] <= step + 10]
    
        ask = instep[instep["side"] == "asks"]
        bid = tenstep[tenstep["side"] == "bids"]
        maxbid = bid["price"].max()
        minask = ask["price"].min()
       # meanbid = bid["price"].mean()
        #meanask = ask["price"].mean()
        
        if (maxbid-minask > 0):
            compteur1 += 1
            objectif.append(1)
        else:
            compteur0 +=  1
            objectif.append(0)
        '''
        if (meanbid-meanask > 0):
            compteur1 += 1
            objectif.append(1)
        else:
            compteur0 += 1
            objectif.append(0)
        '''
    print(compteur1)
    print(compteur0)
    return objectif

#Fonction calculant le taux d'erreur de notre perceptron
def tauxErreur(valeurs, objectif, nb_folds, int_validation):
    somme = 0
    i = int(((len(objectif)-20)/nb_folds)*int_validation)+20
    for j in range (len(valeurs)):
      somme += abs(valeurs[j] - objectif[j+i])
    somme /= len(valeurs)
    return somme*100

#Fonction calculant le taux de précision de notre perceptron
def precision(valeurs, objectif, nb_folds, int_validation):
    somme = 0
    i = int(((len(objectif)-20)/nb_folds)*int_validation)+20
    for j in range (len(valeurs)):
      somme += abs(valeurs[j] - objectif[j+i])
    somme /= len(valeurs)
    return (100-somme*100)
      
#Fonction qui renvoie la base d'apprentissage
def separationBaseApprentissageFolds(donnees_formatees, int_validation, nb_folds):
    i = 0
    base_apprentissage = []
    while (i < len(donnees_formatees)):
        if (i < (len(donnees_formatees)/nb_folds)*int_validation or i > (len(donnees_formatees)/nb_folds)*(int_validation+1)):
            base_apprentissage.append(donnees_formatees[i])
        i += 1
    
    return base_apprentissage

#Fonction qui renvoie la base de validation
def separationBaseValidationFolds(donnees_formatees, int_validation, nb_folds):
    i = int((len(donnees_formatees)/nb_folds)*int_validation)
    base_validation = []
    while (i < (len(donnees_formatees)/nb_folds)*(int_validation+1)):
        base_validation.append(donnees_formatees[i])
        i += 1
    
    return base_validation
