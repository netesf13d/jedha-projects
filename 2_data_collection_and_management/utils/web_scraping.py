#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements web scraping utilities for the project. The functions
implemented are:
- `get_coords`: get geographic coordinates for a selected destination
- ``: get weather data at given geographic coordinates
- ``: get information about hotels in the destination surroundings
"""

import asyncio

import requests


# =============================================================================
# 
# =============================================================================


def get_coords(destination: dict[str, str], **kwargs)-> tuple[float, float]:
    """
    Get the geographic coordinates of a selected destination using
    [Nominatim API](https://nominatim.org/).
    

    Parameters
    ----------
    destination : dict[str, str]
        DESCRIPTION.
    kwargs : str
        Additional arguments passed to Nominatim API.
        

    Returns
    -------
    tuple[float, float]
        The (latitude, longitude) of the destination.
    
    Examples
    --------
    >>> get_coords({'city': "Bayeux", 'country': "France"})
    ()

    """
    params = kwargs | destination | {'format': 'geojson'}
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params=params)
    match r.status_code:
        case 200:
            return filter_results(r.json()['features'])
        case _:
            raise ValueError("request failed")
    


def filter_results(results: list[dict])-> tuple[float, float]:
    """
    

    Parameters
    ----------
    res : list[dict]
        DESCRIPTION.

    Returns
    -------
    tuple[float, float]
        DESCRIPTION.

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
    return tuple(results[0]['geometry']['coordinates'])
        
    

# r = requests.get("https://nominatim.openstreetmap.org/status",
#                   params={'format': "json"})

r = requests.get("https://nominatim.openstreetmap.org/search",
                  params={'city': "Bayeux",
                          'country': "France",
                          'format': "geojson"})

