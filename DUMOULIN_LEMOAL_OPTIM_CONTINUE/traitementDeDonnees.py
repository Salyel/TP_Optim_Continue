import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import perceptron
from perceptron import Perceptron
from traitementDeDonnees import *
import random

from sklearn import datasets
from sklearn.decomposition import PCA

#Fonction formattant les données que l'on va donner au Perceptron.
#Renvoie les différences entre les moyennes bids et les moyennes asks de chaque step et des 20 précédents.
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
        
        for step_davant in range(step-5, step):
            instep_davant = commandes[commandes["step"] == step_davant]
            ask_davant = instep_davant[instep_davant["side"] == "asks"]
            bid_davant = instep_davant[instep_davant["side"] == "bids"]
            meanbid_davant = bid_davant["price"].mean()
            meanask_davant = ask_davant["price"].mean()
            
            donnees_par_step.append(meanbid_actuel-meanbid_davant)
            donnees_par_step.append(meanask_actuel-meanask_davant)
            #donnees_par_step.append((meanbid_actuel-meanbid_davant)+(meanask_actuel-meanask_davant))
            
        donnees_formatees.append(donnees_par_step)
            
    return donnees_formatees

def formatageSur5(commandes):
    donnees_formatees = []
    commandes = commandes[commandes["product_id"] =="BTC-EUR"]
    for step in range(21, 291):
        donnees_par_step = []
        instep = commandes[commandes["step"] == step]
        ask = instep[instep["side"] == "asks"]
        bid = instep[instep["side"] == "bids"]
        meanbid_actuel = bid["price"].mean()
        meanask_actuel = ask["price"].mean()
        
        for step_davant in [step-20, step-15, step-10, step-5]:
            instep_davant = commandes[commandes["step"] == step_davant]
            ask_davant = instep_davant[instep_davant["side"] == "asks"]
            bid_davant = instep_davant[instep_davant["side"] == "bids"]
            meanbid_davant = bid_davant["price"].mean()
            meanask_davant = ask_davant["price"].mean()
            
            donnees_par_step.append(meanbid_actuel-meanbid_davant)
            donnees_par_step.append(meanask_actuel-meanask_davant)
            #donnees_par_step.append((meanbid_actuel-meanbid_davant)+(meanask_actuel-meanask_davant))
            
        donnees_formatees.append(donnees_par_step)
            
    return donnees_formatees

def formatageAvecAvantSur5(commandes):
    donnees_formatees = []
    commandes = commandes[commandes["product_id"] =="BTC-EUR"]
    for step in range(21, 291):
        donnees_par_step = []
        
        for step_davant in [step-20, step-15, step-10, step-5]:
            instep_davant = commandes[commandes["step"] == step_davant]
            instep_dapres = commandes[commandes["step"] == step_davant+5]
            ask_davant = instep_davant[instep_davant["side"] == "asks"]
            bid_davant = instep_davant[instep_davant["side"] == "bids"]
            ask_dapres = instep_dapres[instep_dapres["side"] == "asks"]
            bid_dapres = instep_dapres[instep_dapres["side"] == "bids"]
            meanbid_davant = bid_davant["price"].mean()
            meanask_davant = ask_davant["price"].mean()
            meanbid_dapres = bid_dapres["price"].mean()
            meanask_dapres = ask_dapres["price"].mean()
            
            donnees_par_step.append(meanbid_dapres-meanbid_davant)
            donnees_par_step.append(meanask_dapres-meanask_davant)
            #donnees_par_step.append((meanbid_actuel-meanbid_davant)+(meanask_actuel-meanask_davant))
            
        donnees_formatees.append(donnees_par_step)
            
    return donnees_formatees

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
            donnees_par_step.append(maxbid-minask_davant)
            
        donnees_formatees.append(donnees_par_step)
            
    return donnees_formatees


#Notre fonction qui s'inspire de celle du prof
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
   

#Fonction calculant la précision de notre prédiction en comparant prédiction et objectif
def accuracy(valeurs, objectif):
    compteur_bon = 0
    compteur_faux = 0
    pourcentage = 0
    for i in range (len(valeurs)):
        if (abs(valeurs[i] - objectif[i]) <= 0.01):
            compteur_bon += 1
        else:
            compteur_faux += 1
    pourcentage = compteur_bon/(compteur_bon+compteur_faux)*100
    return pourcentage

#Fonction qui renvoie la base d'apprentissage
def separationBaseApprentissageFolds(donnees_formatees, int_validation, nb_folds):
    i = 0
    base_apprentissage = []
    while (i < len(donnees_formatees)):
        if (i < len(donnees_formatees)/nb_folds*int_validation or i > len(donnees_formatees)/nb_folds*(int_validation+1)):
            base_apprentissage.append(donnees_formatees[i])
        i += 1
    
    return base_apprentissage

#Fonction qui renvoie la base de validation
def separationBaseValidationFolds(donnees_formatees, int_validation, nb_folds):
    i = int(len(donnees_formatees)/nb_folds*int_validation)
    base_validation = []
    while (i < len(donnees_formatees)/nb_folds*(int_validation+1)):
        base_validation.append(donnees_formatees[i])
        i += 1
    
    return base_validation
