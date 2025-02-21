#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import csv

import requests
from requests.adapters import HTTPAdapter, Retry



from etl import get_coords, get_weather_forecast

# from utils.api_calls import get_coords, get_weather



# ## Load API keys
# with open("./utils/api.key", "rt") as f:
#     for line in f.readlines():
#         match line.strip().split(" "):
#             # case ["[OpenWeather]", key]:
#             #     ow_key = key
#             case ["[weatherapi]", key]:
#                 wapi_key = key

with open("./data/locations.csv", 'rt', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader, None) # remove header
    queries = [f"{row[0]}, {row[1]}" for row in reader]


# =============================================================================
# 
# =============================================================================

s = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[403, 502, 503, 504])
s.mount('https://', HTTPAdapter(max_retries=retries))

coords = get_coords(s, queries[4])

# coords = {'latitude': 48.6359541, 'longitude': -1.511459954959514}

# coords = {'longitude': -0.7024738, 'latitude': 49.2764624}
# d = get_weather_forecast(s, **coords)




