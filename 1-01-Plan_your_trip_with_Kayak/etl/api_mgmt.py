# -*- coding: utf-8 -*-
"""
API call to fetch data from open-meteo.
"""

from datetime import datetime, UTC
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests import RequestException



latitude, longitude = 43.330556, 5.354166

features = [
    'time',
    'temperature_2m (°C)',
    'relative_humidity_2m (%)',
    'dew_point_2m (°C)',
    'apparent_temperature (°C)',
    'precipitation (mm)',
    'rain (mm)',
    'snowfall (cm)',
    'snow_depth (m)',
    'weather_code (wmo code)',
    'pressure_msl (hPa)',
    'surface_pressure (hPa)',
    'cloud_cover (%)',
    'cloud_cover_low (%)',
    'cloud_cover_mid (%)',
    'cloud_cover_high (%)',
    'et0_fao_evapotranspiration (mm)',
    'vapour_pressure_deficit (kPa)',
    'wind_speed_10m (km/h)',
    'wind_speed_100m (km/h)',
    'wind_direction_10m (°)',
    'wind_direction_100m (°)',
    'wind_gusts_10m (km/h)',
    'soil_temperature_0_to_7cm (°C)',
    'soil_temperature_7_to_28cm (°C)',
    'soil_temperature_28_to_100cm (°C)',
    'soil_temperature_100_to_255cm (°C)',
    'soil_moisture_0_to_7cm (m³/m³)',
    'soil_moisture_7_to_28cm (m³/m³)',
    'soil_moisture_28_to_100cm (m³/m³)',
    'soil_moisture_100_to_255cm (m³/m³)'
]

hourly_req = [f.split()[0] for f in features[1:]]



# =============================================================================
# 
# =============================================================================

def _from_archive(start_date: str, end_date: str)-> dict:
    """
    Fetch meteo data from archive. Works up to the last 2 days.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
       	"latitude": latitude,
       	"longitude": longitude,
        "start_date": start_date,
       	"end_date": end_date,
       	"hourly": hourly_req,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if 'error' in data:
        raise RequestException(data['reason'])
    return data


def _from_forecast():
    """
    Fetch meteo data from forecast.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
       	"latitude": latitude,
       	"longitude": longitude,
       	"hourly": hourly_req,
         "past_days": 2,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if 'error' in data:
        raise RequestException(data['reason'])
    return data


def fetch_update(last: datetime)-> pd.DataFrame:
    """
    
    """
    now = datetime.now(tz=UTC)
    # recent data from forecast
    data = _from_forecast()
    df = pd.DataFrame(data['hourly']).set_index('time')
    df = df.loc[df.index < now.isoformat()]
    # old data fro archive
    if (now - last).days >= 2:
        data = _from_archive(now.date().isoformat(), now.date.isoformat())
        df_ = pd.DataFrame(data['hourly']).setindex('time')
        df_ = df_.dropna(how='all', axis=0)
        df = df.update(df_)
    
    return df.sort_index()

