# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 3, Question 3
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

import time
import numpy

from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import cdist

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from matplotlib import pyplot


# Fonctions utilitaires liées à l'évaluation
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!") 

# Définition des durées d'exécution maximales pour chaque sous-question
TMAX_FIT = 2.0
TMAX_EVAL = 3.0


# Ne modifiez rien avant cette ligne!


# Question 3B
# Implémentation du discriminant à noyau
class DiscriminantANoyau:

    def __init__(self, lambda_, sigma):
        # Cette fonction est déjà codée pour vous, vous n'avez qu'à utiliser
        # les variables membres qu'elle définit dans les autres fonctions de
        # cette classe.
        # Lambda et sigma sont définis dans l'énoncé.
        self.lambda_ = lambda_
        self.sigma = sigma
    
    def fit(self, X, y):
        # Implémentez la fonction d'entraînement du classifieur, selon
        # les équations que vous avez développées dans votre rapport.

        # TODO Q3B
        # Vous devez écrire une fonction nommée evaluateFunc,
        # qui reçoit un seul argument en paramètre, qui correspond aux
        # valeurs des paramètres pour lesquels on souhaite connaître
        # l'erreur et le gradient d'erreur pour chaque paramètre.
        # Cette fonction sera appelée à répétition par l'optimiseur
        # de scipy, qui l'utilisera pour minimiser l'erreur et obtenir
        # un jeu de paramètres optimal.
       
        def evaluateFunc(hypers):

            return err, grad
        
        # TODO Q3B
        # Initialisez aléatoirement les paramètres alpha et omega0
        # (l'optimiseur requiert un "initial guess", et nous ne pouvons pas
        # simplement n'utiliser que des zéros pour différentes raisons).
        # Stochez ces valeurs initiales aléatoires dans un array numpy nommé
        # "params"
        # Déterminez également les bornes à utiliser sur ces paramètres
        # et stockez les dans une variable nommée "bounds".
        # Indice : les paramètres peuvent-ils avoir une valeur maximale (au-
        # dessus de laquelle ils ne veulent plus rien dire)? Une valeur
        # minimale? Référez-vous à la documentation de fmin_l_bfgs_b
        # pour savoir comment indiquer l'absence de bornes.
       

        # À ce stade, trois choses devraient être définies :
        # - Une fonction d'évaluation nommée evaluateFunc, capable de retourner
        #   l'erreur et le gradient d'erreur pour chaque paramètre pour une
        #   configuration de paramètres alpha et omega_0 donnée.
        # - Un tableau numpy nommé params de même taille que le nombre de
        #   paramètres à entraîner.
        # - Une liste nommée bounds contenant les bornes que l'optimiseur doit 
        #   respecter pour chaque paramètre
        # On appelle maintenant l'optimiseur avec ces informations et on stocke
        # les valeurs optimisées dans params
        _times.append(time.time())
        params, minval, infos = fmin_l_bfgs_b(evaluateFunc, params, bounds=bounds)
        _times.append(time.time())
        checkTime(TMAX_FIT, "Entrainement")

        # On affiche quelques statistiques
        print("Entraînement terminé après {it} itérations et "
                "{calls} appels à evaluateFunc".format(it=infos['nit'], calls=infos['funcalls']))
        print("\tErreur minimale : {:.5f}".format(minval))
        print("\tL'algorithme a convergé" if infos['warnflag'] == 0 else "\tL'algorithme n'a PAS convergé")
        print("\tGradients des paramètres à la convergence (ou à l'épuisement des ressources) :")
        print(infos['grad'])

        # TODO Q3B
        # Stockez les paramètres optimisés de la façon suivante
        # - Le vecteur alpha dans self.alphas
        # - Le biais omega0 dans self.w0



        # On retient également le jeu d'entraînement, qui pourra
        # vous être utile pour les autres fonctions à implémenter
        self.X, self.y = X, y
    
    def predict(self, X):
        # TODO Q3B
        # Implémentez la fonction de prédiction
        # Vous pouvez supposer que fit() a préalablement été exécuté
        # et que les variables membres alphas, w0, X et y existent.
        # N'oubliez pas que ce classifieur doit retourner -1 ou 1

    
    def score(self, X, y):
        # TODO Q3B
        # Implémentez la fonction retournant le score (accuracy)
        # du classifieur sur les données reçues en argument.
        # Vous pouvez supposer que fit() a préalablement été exécuté
        # Indice : réutiliser votre implémentation de predict() réduit de
        # beaucoup la taille de cette fonction!



if __name__ == "__main__":
    # Question 3B

    # TODO Q3B
    # Créez le jeu de données à partir de la fonction make_moons, tel que
    # demandé dans l'énoncé
    # N'oubliez pas de vous assurer que les valeurs possibles de y sont
    # bel et bien -1 et 1, et non 0 et 1!

    
    # TODO Q3B
    # Séparez le jeu de données en deux parts égales, l'une pour l'entraînement
    # et l'autre pour le test
    
    _times.append(time.time())
    # TODO Q3B
    # Une fois les paramètres lambda et sigma de votre classifieur optimisés,
    # créez une instance de ce classifieur en utilisant ces paramètres optimaux,
    # et calculez sa performance sur le jeu de test.


    
    # TODO Q3B
    # Créez ici une grille permettant d'afficher les régions de
    # décision pour chaque classifieur
    # Indice : numpy.meshgrid pourrait vous être utile ici
    # Par la suite, affichez les régions de décision dans la même figure
    # que les données de test.
    # Note : utilisez un pas de 0.02 pour le meshgrid
   
    
    


    # On affiche la figure
    # _times.append(time.time())
    checkTime(TMAX_FIT, "Evaluation")
    pyplot.show()
# N'écrivez pas de code à partir de cet endroit
