# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 3, Question 2
#
###############################################################################
############################## INSTRUCTIONS ###################################
###############################################################################
#
# - Repérez les commentaires commençant par TODO : ils indiquent une tâche que
#       vous devez effectuer.
# - Vous ne pouvez PAS changer la structure du code, importer d'autres
#       modules / sous-modules, ou ajouter d'autres fichiers Python
# - Ne touchez pas aux variables, TMAX*, ERRMAX* et _times, à la fonction
#       checkTime, ni aux conditions vérifiant le bon fonctionnement de votre 
#       code. Ces structures vous permettent de savoir rapidement si vous ne 
#       respectez pas les requis minimum pour une question en particulier. 
#       Toute sous-question n'atteignant pas ces minimums se verra attribuer 
#       la note de zéro (0) pour la partie implémentation!
#
###############################################################################

import itertools
import time
import numpy
import warnings
from io import BytesIO
from http.client import HTTPConnection

from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC

# Nous ne voulons pas avoir ce type d'avertissement, qui
# n'est pas utile dans le cadre de ce devoir
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Fonctions utilitaires liées à l'évaluation
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!") 

# Définition des durées d'exécution maximales pour chaque sous-question
TMAX_KNN = 40
TMAX_SVM = 200
TMAX_PERCEPTRON = 400
TMAX_EVAL = 80

def fetchPendigits():
    """
    Cette fonction télécharge le jeu de données pendigits et le
    retourne sous forme de deux tableaux numpy. Le premier élément
    retourné par cette fonction est un tableau de 10992x16 qui
    contient les samples; le second élément est un vecteur de 10992
    qui contient les valeurs cible (target).
    """
    host = 'vision.gel.ulaval.ca'
    url = '/~cgagne/enseignement/apprentissage/A2018/travaux/ucipendigits.npy'
    connection = HTTPConnection(host, port=80, timeout=10)
    connection.request('GET', url)

    rep = connection.getresponse()
    if rep.status != 200:
        print("ERREUR : impossible de télécharger le jeu de données UCI Pendigits! Code d'erreur {}".format(rep.status))
        print("Vérifiez que votre ordinateur est bien connecté à Internet.")
        return
    stream = BytesIO(rep.read())
    dataPendigits = numpy.load(stream)
    return dataPendigits[:, :-1].astype('float32'), dataPendigits[:, -1]

# Ne modifiez rien avant cette ligne!



if __name__ == "__main__":
    # Question 2B

    # TODO Q2B
    # Chargez le jeu de données Pendigits. Utilisez pour cela la fonction
    # fetchPendigits fournie. N'oubliez pas de normaliser
    # les données d'entrée entre 0 et 1 pour toutes les dimensions.
    # Notez finalement que fetch_openml retourne les données d'une manière
    # différente des fonctions load_*, assurez-vous que vous utilisez
    # correctement les données et qu'elles sont du bon type.
   
    
    # TODO Q2B
    # Séparez le jeu de données Pendigits en deux sous-jeux: entraînement (5000) et
    # test (reste des données). Pour la suite du code, rappelez-vous que vous ne
    # pouvez PAS vous servir du jeu de test pour déterminer la configuration
    # d'hyper-paramètres la plus performante. Ce jeu de test ne doit être utilisé
    # qu'à la toute fin, pour rapporter les résultats finaux en généralisation.


    # TODO Q2B
    # Pour chaque classifieur :
    # - k plus proches voisins,
    # - SVM à noyau gaussien,
    # - Perceptron multicouche,
    # déterminez les valeurs optimales des hyper-paramètres à utiliser.
    # Suivez les instructions de l'énoncé quant au nombre d'hyper-paramètres à
    # optimiser et n'oubliez pas d'expliquer vos choix d'hyper-paramètres
    # dans votre rapport.
    # Vous êtes libres d'utiliser la méthodologie que vous souhaitez, en autant
    # que vous ne touchez pas au jeu de test.
    #
    # Note : optimisez les hyper-paramètres des différentes méthodes dans
    # l'ordre dans lequel ils sont énumérés plus haut, en insérant votre code
    # d'optimisation entre les commentaires le spécifiant
    

    
    _times.append(time.time())
    # TODO Q2B
    # Optimisez ici la paramétrisation du kPP
   

    
    _times.append(time.time())
    checkTime(TMAX_KNN, "K plus proches voisins")
    # TODO Q2B
    # Optimisez ici la paramétrisation du SVM à noyau gaussien
   


    _times.append(time.time())
    checkTime(TMAX_SVM, "SVM")
    # TODO Q2B
    # Optimisez ici la paramétrisation du perceptron multicouche
    # Note : il se peut que vous obteniez ici des "ConvergenceWarning"
    # Ne vous en souciez pas et laissez le paramètre max_iter à sa
    # valeur suggérée dans l'énoncé (100)
    



    _times.append(time.time())
    checkTime(TMAX_PERCEPTRON, "SVM")
    

    # TODO Q2B
    # Évaluez les performances des meilleures paramétrisations sur le jeu de test
    # et rapportez ces performances dans le rapport
    
    

    _times.append(time.time())
    checkTime(TMAX_EVAL, "Evaluation des modèles")
# N'écrivez pas de code à partir de cet endroit
