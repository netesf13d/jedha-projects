# -*- coding: utf-8 -*-
"""
Script for data collection using API calls and web scraping.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from selenium import webdriver

pdir = Path(os.path.abspath('')).resolve().parent
if not (etldir := str(pdir)) in sys.path:
    sys.path.append(etldir)
from etl import (get_coords, get_weather_forecast,
                 scrape_from_searchpage, save_to_json)


# =============================================================================
# API calls
# =============================================================================

# setup session with retry policy in case of failure
s = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[403, 502, 503, 504])
s.mount('https://', HTTPAdapter(max_retries=retries))

# Load locations of interest
with open("../data/places.csv", 'rt', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader, None) # remove header
    places = [row for row in reader]



## Locations information
# get geographic coordinates
coordinates = {}
for i, loc in enumerate(places, start=1):
    coordinates[i] = ({'place': loc[0], 'country': loc[1]} 
                      | get_coords(s, f"{loc[0]}, {loc[1]}"))
    time.sleep(1.1)


# save to csv
locations_cols = ['location_id', 'place', 'country', 'latitude', 'longitude']
with open('../data/locations.csv', 'wt', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writeheader(locations_cols)
    for i, coords in coordinates:
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
    writer = csv.writer(f, delimiter=';')
    writer.writeheader(weather_cols)
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
## https://stackoverflow.com/questions/53039551/selenium-webdriver-modifying-navigator-webdriver-flag-to-prevent-selenium-detec/53040904#53040904
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
with open("../data/locations.csv", 'rt', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader, None) # remove header
    locations = [row for row in reader]

search_urls = {loc[0]: ('https://www.booking.com/searchresults.en-gb.html?'
                        f'latitude={loc[3]}&longitude={loc[4]}')
               for loc in locations}

for i, search_url in search_urls.items():
    hotel_infos = {i: scrape_from_searchpage(driver, search_url, limit=30)} # scrape on search
    save_to_json(f'../data/temp/hotels/{i}.json', hotel_infos)

driver.quit()


"""
The script below scrapes each individual hotel page. This takes a lot of time
and the website is likely to catch selenium as a robot and block it.

```
## open 2 tabs
# search tab
driver.get('https://www.booking.com')
search_window = driver.current_window_handle
# hotel description tab
driver.switch_to.new_window('tab')
driver.get('https://www.booking.com')
hotel_window = driver.current_window_handle

for i, search_url in search_urls.items():
    # get urls of hotel pages
    driver.switch_to.window(search_window)
    hotel_urls = scrape_hotel_urls(driver, search_url) # scrape on search
    hotel_urls = hotel_urls[:30]
    
    # scrape hotel pages
    driver.switch_to.window(hotel_window)
    hotel_infos = []
    for url in hotel_urls:
        time.sleep(0.5)
        try:
            hotel_infos.append(scrape_hotel_info(driver, url))
        except StaleElementReferenceException:
            time.sleep(0.5)
            hotel_infos.append(scrape_hotel_info(driver, url))

    # save intermediate results to json
    save_to_json(f'../data/temp/hotels/{i}.json', hotel_infos)

driver.quit()
```
"""

#%%
# =============================================================================
# Format into csv file and transfer to S3 bucket
# =============================================================================

hotels_list = []
root, _, files = next(os.walk('../data/temp/hotels/'))
for file in files:
    with open(root + file, 'rt', encoding='utf-8') as f:
        hotels_list.append(json.load(f))


hotels_cols = ['hotel_id', 'location_id', 'url', 'name',
               'description', 'rating', 'georating']
hotel_id = 1
hotels_data = []
for hotels in hotels_list:
    loc_id, hotels = next(iter(hotels.values()))
    for h in hotels:
        entry = [hotel_id, loc_id] + [h[col] for col in hotels_cols[2:]]
        hotel_id += 1

with open('../data/hotels.csv', 'wt', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writeheader(hotels_cols)
    for row in hotels_data:
        writer.writerow(row)




# =============================================================================
# Setup database
# =============================================================================




