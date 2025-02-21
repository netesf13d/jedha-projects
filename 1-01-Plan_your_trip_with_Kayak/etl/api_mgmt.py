# -*- coding: utf-8 -*-
"""
API calls to get geocoding and weather forecast.
- `get_coords` fetches the geocoding of named places. We use
  [Nominatim API](https://nominatim.org/).
- `get_weather_forecast` fetches weather forecast at the givzen coordinates.
  We use [Open-Meteo API](https://open-meteo.com/en/docs) to get weather
  forecast information.
"""

import warnings

from requests.sessions import Session
from requests import RequestException

USER_AGENT = 'Mozilla/5.0'

DAILY_WEATHER = ['temperature_2m_max',
                 'temperature_2m_min',
                 'sunshine_duration',
                 'precipitation_sum']


# =============================================================================
# Geocoding information
# =============================================================================

def get_coords(session: Session,
               location: str, **kwargs)-> dict[str, float]:
    """
    Get the geographic coordinates of a selected location using
    [Nominatim API](https://nominatim.org/).
    
    Parameters
    ----------
    session : Session
        HTTP session. Used to persist some behavior such as retry.
    location : str
        Location to look for geocoding.
    kwargs : str
        Additional arguments passed to Nominatim API.
        
    Returns
    -------
    dict[str, float]
        The {'lat': latitude, 'lon': longitude} of the location.
    
    Examples
    --------
    >>> import requests
    >>> s = requests.Session()
    >>> get_coords('Bayeux, France')
    {'lat': 49.2764624, 'lon': -0.7024738} # may also raise HTTPError
    >>> get_coords('idontknow, abcdefgh')
    {'lat': nan, 'lon': nan} # may also raise HTTPError
    """
    params = kwargs | {'q': location, 'format': 'geojson'}
    r = session.get("https://nominatim.openstreetmap.org/search",
                     headers={'User-agent': USER_AGENT},
                     params=params)
    r.raise_for_status()
    results = r.json()['features']
    if not results:
        msg = f"Queried location {location} dit not return any result"
        warnings.warn(msg)
        return {'lat': float('nan'), 'lon': float('nan')}
    return _filter_results(results)
    

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
    return {'latitude': coords[1], 'longitude': coords[0]}


# =============================================================================
# Weather forecast
# =============================================================================

def get_weather_forecast(session: Session,
                         latitude: float, longitude: float,
                         daily_data: list[str] = DAILY_WEATHER,
                         )-> dict:
    """
    Get weather forecast at the requested geographic coordinates using
    [Open-Meteo](https://open-meteo.com/en/docs).
    
    Parameters
    ----------
    session : Session
        HTTP session. Used to persist some behavior such as retry.
    latitude, longitude : float
        Geographic coordinates.
    daily_data : list[str]
        Daily weather variables to fetch.
        See https://open-meteo.com/en/docs for detailed information.

    Returns
    -------
    dict
        Weather information.
        
    Examples
    --------
    >>> import requests
    >>> s = requests.Session()
    >>> coords = {'longitude': -0.7024738, 'latitude': 49.2764624} # Bayeux
    >>> get_weather_forecast(s, **coords)
    {'latitude': 49.28,
     'longitude': -0.70000005,
     'generationtime_ms': 0.21529197692871094,
     'utc_offset_seconds': 3600,
     'timezone': 'Europe/Paris',
     'timezone_abbreviation': 'GMT+1',
     'elevation': 49.0,
     'daily_units': {'time': 'iso8601',
     'temperature_2m_max': '°C',
     'temperature_2m_min': '°C',
     'sunshine_duration': 's',
     'precipitation_sum': 'mm'},
     'daily': {'time': ['2025-02-21', ..., '2025-03-08'],
     'temperature_2m_max': [15.1,..., 8.5],
     'temperature_2m_min': [11.5,..., 4.2],
     'sunshine_duration': [2452.24, ..., 37531.61],
     'precipitation_sum': [1.5, ... 0.0]}}
    """
    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
       	'latitude': latitude,
       	'longitude': longitude,
        'daily': daily_data,
        # 'hourly': hourly_data,
        'timezone': 'auto',
        'forecast_days': 16,
    }
    r = session.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if 'error' in data:
        raise RequestException(data['reason'])
    return data