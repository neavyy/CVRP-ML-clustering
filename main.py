import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_delivery_points(nombre_points,a=15, b=15,graine=None):
    """Renvoie une liste de tuples (i, j) distincts tires uniformement.

    La grille contient a * b positions : 0 <= i < a et 0 <= j < b.
    Une graine permet de reproduire le tirage sans modifier l'aleatoire global.
    """
    for nom, valeur in (("a", a), ("b", b), ("nombre_points", nombre_points)):
        if isinstance(valeur, bool) or not isinstance(valeur, int):
            raise TypeError(f"{nom} doit etre un entier.")
    if a <= 0 or b <= 0:
        raise ValueError("Les dimensions a et b doivent etre strictement positives.")
    if not 0 <= nombre_points <= a * b:
        raise ValueError("nombre_points doit etre compris entre 0 et a * b.")

    generateur = random.Random(graine)
    positions = generateur.sample(range(a * b), nombre_points)
    return [divmod(position, b) for position in positions]


def generate_depot_and_delivery_points(nombre_points, a=15, b=15, graine=None):
    """Renvoie (depot, points_livraison), sans aucune case partagee.

    Le depot est un tuple (i, j), les livraisons une liste de tuples.
    nombre_points compte uniquement les livraisons : il faut donc
    nombre_points + 1 cases disponibles dans la grille.
    """
    if isinstance(nombre_points, bool) or not isinstance(nombre_points, int):
        raise TypeError("nombre_points doit etre un entier.")
    if nombre_points < 0:
        raise ValueError("nombre_points doit etre positif ou nul.")

    # Un tirage commun sans repetition reserve une case distincte au depot.
    points = generate_delivery_points(nombre_points + 1, a, b, graine)
    depot = points[0]
    points_livraison = points[1:]
    return depot, points_livraison


def generate_distance_matrix(depot, points_livraison):
    """Renvoie les distances Manhattan avec une diagonale infinie.

    Les lignes et colonnes suivent l'ordre [depot] + points_livraison.
    Chaque ligne commence par la distance au depot (infini pour le depot).
    math.inf penalise le choix du point courant comme prochaine destination.
    Si tous les candidats ont un cout infini, aucun deplacement n'est possible.
    """
    positions = [depot] + list(points_livraison)
    return [
        [
            math.inf if origine == destination else abs(i1 - i2) + abs(j1 - j2)
            for destination, (i2, j2) in enumerate(positions)
        ]
        for origine, (i1, j1) in enumerate(positions)
    ]


def validate_cvrp_inputs(distances, demandes, capacite, nb_vehicules):
    """Verifie la matrice, les demandes et la capacite totale de la flotte."""
    n = len(distances)
    if n == 0 or any(len(ligne) != n for ligne in distances):
        raise ValueError("distances doit etre une matrice carree incluant le depot.")
    if len(demandes) != n or demandes[0] != 0:
        raise ValueError("Il faut une demande par point, dont zero pour le depot.")
    if not math.isfinite(capacite) or capacite <= 0:
        raise ValueError("capacite doit etre finie et strictement positive.")
    if any(not math.isfinite(q) or q < 0 or q > capacite for q in demandes):
        raise ValueError("Chaque demande doit etre finie, positive ou nulle et <= capacite.")
    if isinstance(nb_vehicules, bool) or not isinstance(nb_vehicules, int):
        raise TypeError("nb_vehicules doit etre un entier.")
    if nb_vehicules < 0:
        raise ValueError("nb_vehicules doit etre positif ou nul.")
    if n > 1 and nb_vehicules == 0:
        raise ValueError("Il faut au moins un vehicule pour desservir les livraisons.")
    demande_totale = sum(demandes)
    if capacite * nb_vehicules < demande_totale:
        raise ValueError(
            f"Flotte insuffisante : {capacite} * {nb_vehicules} "
            f"= {capacite * nb_vehicules} < demande totale {demande_totale}."
        )
    for i in range(n):
        for j in range(i + 1, n):
            if (not math.isfinite(distances[i][j])
                    or distances[i][j] < 0
                    or distances[i][j] != distances[j][i]):
                raise ValueError("Les distances hors diagonale doivent etre finies, positives ou nulles et symetriques.")


def calculate_savings(distances):
    """Renvoie les tuples (economie, i, j), tries par economie decroissante."""
    n = len(distances)
    # Relier i a j supprime les deux arcs vers le depot et ajoute l'arc i -> j.
    economies = [
        (distances[i][0] + distances[0][j] - distances[i][j], i, j)
        for i in range(1, n)
        for j in range(i + 1, n)
    ]
    economies.sort(key=lambda element: (-element[0], element[1], element[2]))
    return economies


def merge_routes(economies, demandes, capacite):
    """Construit et fusionne les routes ; renvoie leurs clients sans le depot.

    Les economies doivent etre triees et les entrees validees au prealable.
    Les structures routes, charges et appartenance restent locales.
    """
    n = len(demandes)
    # Initialement, chaque livraison possede sa propre tournee 0 -> i -> 0.
    routes = {i: [i] for i in range(1, n)}
    charges = {i: demandes[i] for i in range(1, n)}
    appartenance = {i: i for i in range(1, n)}

    for economie, i, j in economies:
        if economie < 0:
            break
        ri, rj = appartenance[i], appartenance[j]
        if ri == rj or charges[ri] + charges[rj] > capacite:
            continue
        route_i, route_j = routes[ri], routes[rj]
        # Une fusion n'est possible qu'aux extremites de deux routes distinctes.
        if i not in (route_i[0], route_i[-1]) or j not in (route_j[0], route_j[-1]):
            continue
        if route_i[-1] != i:
            route_i = route_i[::-1]
        if route_j[0] != j:
            route_j = route_j[::-1]

        routes[ri] = route_i + route_j
        charges[ri] += charges.pop(rj)
        for client in route_j:
            appartenance[client] = ri
        del routes[rj]
    return list(routes.values())


def calculate_total_distance(tournees, distances):
    """Additionne les arcs des tournees, qui incluent deja le depot aux bouts."""
    return sum(
        distances[i][j]
        for tournee in tournees
        for i, j in zip(tournee, tournee[1:])
    )


def clarke_wright(distances, demandes, capacite, nb_vehicules):
    """Construit des tournees par Clarke et Wright (version parallele).

    distances : matrice symetrique, depot a l'indice 0 ; diagonale ignoree.
    demandes : quantites a livrer, dans le meme ordre que la matrice.
        demandes[0] = 0 pour le depot ; demandes[i] concerne la livraison i.
        Exemple : [0, 2, 1] signifie 2 unites au client 1 et 1 au client 2.
        Pour une unite par livraison : [0] + [1] * (len(distances) - 1).
    capacite : charge maximale par vehicule, dans la meme unite que demandes.
    nb_vehicules : taille de flotte (entier positif ou nul), une tournee
        au maximum par vehicule ; les livraisons ne sont pas fractionnees.

    Renvoie (tournees, distance_totale). Chaque tournee contient les indices
    des points, avec 0 au debut et a la fin.
    Verifie capacite * nb_vehicules >= sum(demandes), soit le nombre de
    livraisons hors depot lorsque chaque demande vaut 1.
    Cette condition est necessaire, mais ne garantit pas une solution.
    Leve RuntimeError si l'heuristique produit plus de tournees que de
    vehicules, sans conclure que le probleme est impossible.
    Les fusions de gain nul sont acceptees pour reduire le nombre de tournees.
    C'est une heuristique : l'optimalite n'est pas garantie.
    """
    validate_cvrp_inputs(distances, demandes, capacite, nb_vehicules)
    economies = calculate_savings(distances)
    routes = merge_routes(economies, demandes, capacite)

    tournees = [[0] + route + [0] for route in routes]
    if len(tournees) > nb_vehicules:
        raise RuntimeError(
            f"Clarke et Wright a produit {len(tournees)} tournees pour "
            f"{nb_vehicules} vehicules. Aucune solution respectant la flotte "
            "n'a ete trouvee par cette heuristique ; cela ne prouve pas "
            "que le probleme est impossible."
        )
    distance_totale = calculate_total_distance(tournees, distances)
    return tournees, distance_totale
