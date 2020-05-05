
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import perceptron
from perceptron import Perceptron

from sklearn import datasets
from sklearn.decomposition import PCA

def remplissageObjectif10SuivantsMinimumMaximum(donnees):
    i = 0
    derniereValeurStep = 0
    objectif = []
    for k in range(1, 291):
        i = derniereValeurStep
        step = donnees[i][5]
        augmentation = 0
        minimum = 9999999999
        #On récupère la plus petite valeur des offres de notre step
        while (donnees[i][5] == step):
            if (donnees[i][3] == 'bids') and (donnees[i][2] == 'BTC-EUR'):
                if donnees[i][4] < minimum:
                    minimum = donnees[i][4]
            i += 1
        derniereValeurStep = i
        while augmentation == 0 and donnees[i][5] <= step+10:    #Tant qu'on sait pas encore si ça a augmenté, et qu'on est pas 10 steps plus loin
            if (donnees[i][3] == 'asks') and (donnees[i][2] == 'BTC-EUR'):     #Si c'est une demande et que c'est le même type d'échange
                if (donnees[i][4] > minimum):   #Si la demande est plus élevée que le prix de l'offre
                    augmentation = 1#donnees[i][4] - minimum
            i += 1
        objectif.append(augmentation)
    return objectif

def remplissageObjectif10SuivantsMoyenne(donnees):
    i = 0
    derniereValeurStep = 0
    objectif = []
    for k in range(1, 291):
        i = derniereValeurStep
        step = donnees[i][5]
        augmentation = 0
        somme = 0
        compteur = 0
        somme10 = 0
        compteur10 = 0

        while (donnees[i][5] == step):
            if (donnees[i][3] == 'bids') and (donnees[i][2] == 'BTC-EUR'):
                somme += donnees[i][4]
                compteur += 1
            i += 1
        somme /= compteur
        derniereValeurStep = i
        while donnees[i][5] <= step+10 and i < 89999:    #Tant qu'on sait pas encore si ça a augmenté, et qu'on est pas 10 steps plus loin
            if (donnees[i][3] == 'asks') and (donnees[i][2] == 'BTC-EUR'):     #Si c'est une demande et que c'est le même type d'échange
                somme10 += donnees[i][4]
                compteur10 += 1
            i += 1
        somme10 /= compteur10
        if (somme < somme10):
            augmentation = somme10 - somme
        objectif.append(augmentation)
    return objectif

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
        
        for step_davant in range(step-20, step):
            instep_davant = commandes[commandes["step"] == step_davant]
            ask_davant = instep_davant[instep_davant["side"] == "asks"]
            bid_davant = instep_davant[instep_davant["side"] == "bids"]
            meanbid_davant = bid_davant["price"].mean()
            meanask_davant = ask_davant["price"].mean()
            
            donnees_par_step.append(meanbid_actuel-meanbid_davant)
            donnees_par_step.append(meanask_actuel-meanask_davant)
            
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
        
        if (maxbid-minask > 0):
            compteur1 += 1
            objectif.append(1)
        else:
            compteur0 +=  1
            objectif.append(0)
    print(compteur1)
    print(compteur0)
    return objectif
   
##Fonction du prof qui marche     
def fonctionProf():
    commandes = pd.read_csv('ask_bid.csv', sep=";")
    commandes = commandes[commandes["product_id"] =="BTC-EUR"]
    commandes["increase"] = commandes["price"]
    variation = []
    coef = 1.0
    val = 0
    nbn = 0
    nbp = 0
    for step in range(1,291):
        instep = commandes[commandes["step"] == step]
        tenstep = commandes[commandes["step"] > step]
        tenstep = tenstep[tenstep["step"] <= step + 10]
    
        ask = instep[instep["side"] == "asks"]
        bid = tenstep[tenstep["side"] == "bids"]
        #instep["value"] = instep["price"]instep["size"]
        maxbid = bid["price"].max()
        #weigthedMeanbid = instep["value"].sum()/instep["size"].sum()
        #maxask = tenstep["price"].max()
        #tenstep["value"] = tenstep["price"]tenstep["size"]
        #weigthedMeanask = tenstep["value"].sum()/tenstep["size"].sum()
        minask = ask["price"].min()
        #print(weigthedMeanask-weigthedMeanbid)
        variation.append(maxbid-minask)
        #print(maxbid-coef*minask - val)
        if (maxbid-minask < 0):
            nbn += 1
        else:
            nbp += 1
    
    plt.plot(variation)
    plt.show()
    print(nbn, nbp)
    plt.boxplot(variation)
    plt.show()
    commandes = commandes[commandes["step"] < 291]
    commandes["class"] = 1
    for step in range(290):
        if(variation[step] <= 0):
            commandes.loc[commandes.step == step+1, 'class'] = 0
    
    commandes

#Fonction calculant la précision de notre prédiction en comparant prédiction et objectif
def accuracy(valeurs, objectif):
    compteur_bon = 0
    compteur_faux = 0
    pourcentage = 0
    for i in range (len(valeurs)):
        if (valeurs[i] - objectif[i] <= 0.1):
            compteur_bon += 1
        else:
            compteur_faux += 1
    pourcentage = compteur_bon/(compteur_bon+compteur_faux)*100
    return pourcentage

donnees_brutes = pd.read_csv('ask_bid.csv', sep=";")
objectif = []
objectif = remplissage10SuivantsProf(donnees_brutes)
donnees_formatees = []
donnees_formatees = formatage(donnees_brutes)
#print(donnees_brutes)
donnees = donnees_brutes.to_numpy()
lambdda = 0.1

for i in range (len(donnees_formatees)):
    donnees_formatees[i].append(1)
    
perceptron = Perceptron(40)
for i in range (200):
    print("// Itération", i)
    valeur = []
    valeur = perceptron.calculFonctionEntrainement(donnees_formatees, objectif, lambdda)
    pourcentage = accuracy(valeur, objectif)
    print("Le pourcentage de précision est de ", pourcentage)

'''
Pour la prochaine fois : Faire en sorte de répartir des données d'apprentissage pour que ce soit pas toujours
dans le même ordre, ou bien faire base apprentissage / base de test / base de validation.
'''
    
#print(donnees)
#print(objectif)
#print(donnees_formatees)


'''
Les données qu'on utilise ça va être le price:
    - couper en pleins de morceaux (en 3 par 3 de step), et on regarde la diff entre les évolutions des différences
    des moyennes
    
    - (moyenne des différences à la moyenne de notre step - moyenne des différences à la moyenne de chaque autre step)
'''

