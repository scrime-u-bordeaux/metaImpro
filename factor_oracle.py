import os
import json
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Union


"""
Ce code contient trois fonctions :
- Une pour comparer les symboles entre eux,
- La seconde pour renvoyer l'indice d'un symbole similaire dans un dictionnaire,
- La dernière qui permet de construire l'oracle  à partir des symboles
- En bonus, une fonction pour calculer le rsl si cela est utile plus tard


"""

def symbols_are_similar(sym1, sym2):
    """
    Compare deux symboles  en vérifiant l'égalité de chacun de leurs éléments.
    Renvoie True si tous les éléments sont égaux.
    """
    return sym1[0] == sym2[0] and sym1[1] == sym2[1] and sym1[2] == sym2[2]


def find_similar(d, sigma):
    """
    Parcourt le dictionnaire d transitions (d) pour vérifier si une clé déjà présente est similaire à sigma.
    Renvoie la clé existante si trouvée, sinon retourne None.
    """
    for key in d:
        if symbols_are_similar(key, sigma):
            return key
    return None


def oracle(sequence):
    """
    Construit un oracle des facteurs à partir d'une séquence de symboles.
    """
    transitions = {0: {}}
    supply = {0: -1}
    currentState = 0  # dernier état créé (initialement 0)

    def addSymbol(sigma, m, transitions, supply):
        # Convertir le symbole en tuple s'il est passé sous forme de liste
        if isinstance(sigma, list):
            sigma = tuple(sigma)

        newState = m + 1
        transitions[newState] = {}  # Nouveau dictionnaire pour l'état newState

        # Utilisation de find_similar pour vérifier s'il existe déjà une clé similaire dans transitions[m]
        if find_similar(transitions[m], sigma) is None:
            transitions[m][sigma] = newState
        else:
            # Même si un symbole similaire existe déjà, on écrase la transition pour forcer le lien avec newState
            transitions[m][sigma] = newState

        k = supply[m]
        while k > -1 and find_similar(transitions[k], sigma) is None:
            transitions[k][sigma] = newState
            k = supply[k]

        if k == -1:
            s = 0
        else:
            key_match = find_similar(transitions[k], sigma)
            s = transitions[k][key_match]
        supply[newState] = s

        return newState

    # Lecture de la séquence symbole par symbole
    for symbol in sequence:
        currentState = addSymbol(symbol, currentState, transitions, supply)

    return transitions, supply

def compute_rsl(supply, sequence):
    """
    Calcule la longueur du contexte répété pour chaque état.
    
    Args:
        supply: dictionnaire des suffix links
        sequence: séquence originale
    
    Returns:
        Un tableau des longueurs de contexte
    """
    n = len(sequence)
    rsl = [0] * n
    
    for i in range(1, n):
        if supply[i] != -1:
            j = supply[i]
            if j > 0:
                # La longueur du contexte est la longueur du plus long suffixe répété
                rsl[i] = rsl[j] + 1
            else:
                # Si le suffixe pointe vers l'état initial, la longueur est 1
                rsl[i] = 1
    
    return rsl