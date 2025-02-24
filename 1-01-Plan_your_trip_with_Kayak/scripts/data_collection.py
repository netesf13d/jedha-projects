# -*- coding: utf-8 -*-
"""
Script for data collection and storage.
"""

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from selenium import webdriver
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import Session

pdir = Path(os.path.abspath('')).resolve().parent
if not (etldir := str(pdir)) in sys.path:
    sys.path.append(etldir)
from etl import (get_coords, get_weather_forecast,
                 scrape_from_searchpage,
                 save_to_json,
                 download_csv,
                 Base, Location, Hotel, WeatherIndicator)


BUCKET_NAME = '../bucket_name.key'
S3_WRITER_ACCESS_KEYS = '../jedha-project-s3-writer_accessKeys.key'
S3_READER_ACCESS_KEYS = '../jedha-project-s3-reader_accessKeys.key'
NEONDB_ACCESS_KEYS = '../neondb_access_keys.key'


# =============================================================================
# API calls
# =============================================================================

# setup session with retry policy in case of failure
s = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[403, 502, 503, 504])
s.mount('https://', HTTPAdapter(max_retries=retries))

# Load locations of interest
with open("./data/places.csv", 'rt', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader, None) # remove header
    locations = [row for row in reader]



## Locations information
# get geographic coordinates
coordinates = {}
for i, loc in enumerate(locations, start=1):
    coordinates[i] = ({'name': loc[0], 'country': loc[1]}
                      | get_coords(s, f"{loc[0]}, {loc[1]}"))
    time.sleep(1.1)


# save to csv
locations_cols = ['location_id', 'name', 'country', 'latitude', 'longitude']
with open('../data/locations.csv', 'wt', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_STRINGS)
    writer.writerow(locations_cols)
    for i, coords in coordinates.items():
        writer.writerow([i] + [coords[col] for col in locations_cols[1:]])



## weather forecast
# get weather forecast
weather_forecast = {}
for i, coords in coordinates.items():
    weather_forecast[i] = get_weather_forecast(
        s, coords['latitude'], coords['longitude'])


# save to csv
# take the average weather over the next 7 days
weather_cols = [
    'location_id', 'date', 'min_temperature_C', 'max_temperature_C',
    'sunshine_duration_h', 'precipitation_sum_mm'
]
with open('../data/weather_indicators.csv', 'wt', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_STRINGS)
    writer.writerow(weather_cols)
    for i, forecast in weather_forecast.items():
        forecast = forecast['daily']
        row = [i, forecast['time'][0],
               np.mean(forecast['temperature_2m_min'][:8]),
               np.mean(forecast['temperature_2m_max'][:8]),
               np.mean(forecast['sunshine_duration'][:8])/3600,
               np.mean(forecast['precipitation_sum'][:8])]
        writer.writerow(row)



# =============================================================================
# Web scraping
# =============================================================================

## setup driver with options to prevent detection
## See https://stackoverflow.com/questions/53039551/selenium-webdriver-modifying-navigator-webdriver-flag-to-prevent-selenium-detec/53040904#53040904
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options)
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd(
    'Network.setUserAgentOverride',
    {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'}
)
driver.implicitly_wait(1)


# Load locations of interest
with open("./data/locations.csv", 'rt', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_STRINGS)
    next(reader, None) # remove header
    locations = [row for row in reader]

search_urls = {int(loc[0]): ('https://www.booking.com/searchresults.en-gb.html?'
                            f'latitude={loc[3]}&longitude={loc[4]}')
               for loc in locations}

for i, search_url in search_urls.items():
    hotel_infos = {i: scrape_from_searchpage(driver, search_url, limit=30)} # scrape on search
    save_to_json(f'../data/temp/{i}.json', hotel_infos)

driver.quit()


# load temporary saved files
hotels_list = []
root, _, files = next(os.walk('../data/temp/'))
for file in files:
    with open(root + file, 'rt', encoding='utf-8') as f:
        hotels_list.append(json.load(f))

# transform and save csv
hotels_cols = ['hotel_id', 'location_id', 'url', 'name',
               'description', 'rating', 'georating']
hotel_id = 1
hotels_data = []
for hotels in hotels_list:
    loc_id, hotels = next(iter(hotels.items()))
    for h in hotels:
        entry = [hotel_id, int(loc_id)] + [h[col] for col in hotels_cols[2:]]
        hotels_data.append(entry)
        hotel_id += 1

with open('../data/hotels.csv', 'wt', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_STRINGS)
    writer.writerow(hotels_cols)
    for row in hotels_data:
        writer.writerow(row)


## remove temp directory
shutil.rmtree('../data/temp/')


# =============================================================================
# Storage in a data lake (S3 bucket)
# =============================================================================

## create S3 client with writing permission
with open(BUCKET_NAME, 'rt', encoding='utf-8') as f:
    bucket_name = f.read()
with open(S3_WRITER_ACCESS_KEYS, 'rt', encoding='utf-8') as f:
    aws_access_key_id, aws_secret_access_key = f.readlines()[-1].strip().split(',')

s3_writer = boto3.client('s3', # region_name=region_name,
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key)


## Uplpoad the files created
s3_writer.upload_file('../data/locations.csv', Bucket=bucket_name, Key='data/locations.csv')
s3_writer.upload_file('../data/weather_indicators.csv', Bucket=bucket_name, Key='data/weather_indicators.csv')
s3_writer.upload_file('../data/hotels.csv', Bucket=bucket_name, Key='data/hotels.csv')


# =============================================================================
# Storage in a database (PostgreSQL in Neon DB)
# =============================================================================

## get credentials
with open(NEONDB_ACCESS_KEYS, 'rt', encoding='utf-8') as f:
    pghost = f.readline().split("'")[1]
    pgdatabase = f.readline().split("'")[1]
    pguser = f.readline().split("'")[1]
    pgpassword = f.readline().split("'")[1]

url = URL.create(
    "postgresql+psycopg",
    username=pguser,
    password=pgpassword,
    host=pghost,
    database=pgdatabase,
)

## setup SQL engine
engine = create_engine(url, echo=False)
# inspector = inspect(engine)

## clear the database
# Base.metadata.drop_all(engine)

## create the database
Base.metadata.create_all(engine)


## create S3 client with only readig permission
with open(S3_READER_ACCESS_KEYS, 'rt', encoding='utf-8') as f:
    aws_access_key_id, aws_secret_access_key = f.readlines()[-1].strip().split(',')

s3_reader = boto3.client('s3', # region_name=region_name,
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key)


## Load data from S3 buckets
locations = download_csv(s3_reader, bucket_name, 'data/locations.csv')
locations = [Location(**d) for d in locations]

weather_indicators = download_csv(s3_reader, bucket_name, 'data/weather_indicators.csv')
weather_indicators = [WeatherIndicator(**d) for d in weather_indicators]

hotels = download_csv(s3_reader, bucket_name, 'data/hotels.csv')
hotels = [Hotel(**d) for d in hotels]


## Transfer to database
Base.metadata.create_all(engine)
with Session(engine) as session:
    session.add_all(locations)
    session.commit()
    session.add_all(weather_indicators)
    session.commit()
    session.add_all(hotels)
    session.commit()

## close database connection gracefully
engine.dispose()