#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import csv

from utils.api_calls import get_coords, get_weather



## Load API keys
with open("./utils/api.key", "rt") as f:
    for line in f.readlines():
        match line.strip().split(" "):
            # case ["[OpenWeather]", key]:
            #     ow_key = key
            case ["[weatherapi]", key]:
                wapi_key = key



# =============================================================================
# 
# =============================================================================

locations = ["Mont Saint Michel",
"St Malo",
"Bayeux",
"Le Havre",
"Rouen",
"Paris",
"Amiens",
"Lille",
"Strasbourg",
"Chateau du Haut Koenigsbourg",
"Colmar",
"Eguisheim",
"Besancon",
"Dijon",
"Annecy",
"Grenoble",
"Lyon",
"Gorges du Verdon",
"Bormes les Mimosas",
"Cassis",
"Marseille",
"Aix en Provence",
"Avignon",
"Uzes",
"Nimes",
"Aigues Mortes",
"Saintes Maries de la mer",
"Collioure",
"Carcassonne",
"Ariege",
"Toulouse",
"Montauban",
"Biarritz",
"Bayonne",
"La Rochelle"]


with open("./locations.csv", 'wt', encoding='utf-8') as f:
    f.write("place,country\n")
    for loc in locations:
        f.write(f"{loc},France\n")


with open("./locations.csv", 'rt', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader, None) # remove header
    queries = [f"{row[0]}, {row[1]}" for row in reader]

# a = get_coords(locations[9])


# =============================================================================
# 
# =============================================================================

coordinates = {q: {'lon': -0.7024738, 'lat': 49.2764624} for q in queries}


"Chateau du Haut Koenigsbourg, Orschwiller"




