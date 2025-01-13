#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements web scraping utilities for the project. The functions
implemented are:
- `get_coords`: get geographic coordinates for a selected location
- ``: get weather data at given geographic coordinates
- ``: get information about hotels in the location surroundings
"""

import asyncio
import warnings

import requests


# =============================================================================
# 
# =============================================================================

def get_coords(location: dict[str, str], **kwargs)-> dict[str, float]:
    """
    Get the geographic coordinates of a selected location using
    [Nominatim API](https://nominatim.org/).
    
    Parameters
    ----------
    location : dict[str, str]
        DESCRIPTION.
        'amenity': name and/or type of POI
        'street': housenumber and streetname
        'city': city
        'county': county
        'state': state
        'country': country
        'postalcode': postal code
    kwargs : str
        Additional arguments passed to Nominatim API.
        
    Returns
    -------
    dict[str, float]
        The {'lat': latitude, 'lon': longitude} of the location.
    
    Examples
    --------
    >>> get_coords({'city': "Bayeux", 'country': "France"})
    {'lat': 49.2764624, 'lon': -0.7024738} # may also raise ValueError("access blocked")
    >>> get_coords({'city': "idontknow", 'country': "nullsurrrrrrrrrrrrr"})
    {'lat': nan, 'lon': nan} # may also raise ValueError("access blocked")
    """
    params = kwargs | location | {'format': 'geojson'}
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params=params)
    match r.status_code:
        case 200:
            results = r.json()['features']
            if not results:
                msg = f"Queried location {location} dit not return any result"
                warnings.warn(msg)
                return {'lat': float('nan'), 'lon': float('nan')}
            return _filter_results(results)
        case 403:
            raise ValueError("403: access blocked")
        case _:
            raise ValueError(f"problem: {r.content()}")
    

def _filter_results(results: list[dict])-> tuple[float, float]:
    """
    Filter multiple location results returned by a request to nominatim to get
    the most relevant.
    """
    favorite_address_types = [
        'city',
        'town',
    ]
    for address_type in favorite_address_types:
        for res in results:
            if (res['type'] == 'administrative'
                and res['addresstype'] == address_type):
                return tuple(res['geometry']['coordinates'])
    # default to the first result
    coords = results[0]['geometry']['coordinates']
    return {'lat': coords[1], 'lon': coords[0]}
        

def get_weather(api_call: str,
                coords: dict[str, float],
                api_key: str,
                **kwargs: str)-> dict:
    """
    Get weather information at the requested geographic coordinates using
    [weatherapi](https://www.weatherapi.com/).

    Parameters
    ----------
    api_call : str
        API call, eg 'current.json' or 'forcast.json'.
    coords : dict[str, float]
        Geographic coordinates.
    api_key : str
        API key.
    **kwargs : str
        Additional parameters passed to the call.

    Returns
    -------
    dict
        Weather information.
        
    Examples
    --------
    >>> coords = {'lon': -0.7024738, 'lat': 49.2764624} # Bayeux
    >>> get_weather('current.json', coords, weatherapi_key)
    {'location': {'name': 'Bayeux',
      'region': 'Nord-Pas-de-Calais',
      'country': 'France',
      ...
      'temp_c': 15.0,
      'temp_f': 59.0,
      'is_day': 1,
      'condition': {'text': 'Partly cloudy',
       'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png',
       'code': 1003},
      ...
    }
    
    """
    params = kwargs | {'q': f"{coords['lat']},{coords['lon']}", 'key': api_key}
    r = requests.get(
        f"https://api.weatherapi.com/v1/{api_call}",
        params=params)
    match r.status_code:
        case 200:
            return r.json()
        case 400:
            raise ValueError("bad request")
        case 401:
            raise ValueError("unauthorized access; go figure...")
        case 403:
            raise ValueError("access blocked")
        case _:
            raise ValueError(f"problem: {r.content()}")



# params = {'q': "Chateau du Haut Koenigsbourg, France"} | {'format': 'geojson'}
# r = requests.get("https://nominatim.openstreetmap.org/search",
#                  params=params)
# print(r)
"""
{'type': 'FeatureCollection',
 'licence': 'Data © OpenStreetMap contributors, ODbL 1.0. http://osm.org/copyright',
 'features': [{'type': 'Feature',
   'properties': {'place_id': 394724560,
    'osm_type': 'way',
    'osm_id': 1299451538,
    'place_rank': 30,
    'category': 'historic',
    'type': 'castle',
    'importance': 0.3810240986836976,
    'addresstype': 'historic',
    'name': 'Château du Haut-Kœnigsbourg',
    'display_name': 'Château du Haut-Kœnigsbourg, D 159, Château du Haut-Kœnigsbourg, Orschwiller, Sélestat-Erstein, Bas-Rhin, Grand Est, France métropolitaine, 67600, France'},
   'bbox': [7.3425172, 48.2489931, 7.346206, 48.2498214],
   'geometry': {'type': 'Point',
    'coordinates': [7.344320233724503, 48.249410749999996]}},
  {'type': 'Feature',
   'properties': {'place_id': 104591830,
    'osm_type': 'node',
    'osm_id': 4245068168,
    'place_rank': 22,
    'category': 'place',
    'type': 'isolated_dwelling',
    'importance': 0.10672799038681618,
    'addresstype': 'isolated_dwelling',
    'name': 'Château du Haut-Kœnigsbourg',
    'display_name': 'Château du Haut-Kœnigsbourg, Orschwiller, Sélestat-Erstein, Bas-Rhin, Grand Est, France métropolitaine, 67600, France'},
   'bbox': [7.3454423, 48.2494726, 7.3455423, 48.2495726],
   'geometry': {'type': 'Point', 'coordinates': [7.3454923, 48.2495226]}}]}
"""

params = {'q': "Paris, France"} | {'format': 'geojson'}
r = requests.get("https://nominatim.openstreetmap.org/search",
                 params=params)
print(r)
