# -*- coding: utf-8 -*-
"""
This package implement Extract-Transform-Load helper functions.
- `utils`: utilities of the package
- `api_mgmt`: API calls to fetch geocoding and weather forecast
- `scraping_mgmt`: utilities for web scraping of booking.com pages
- `s3_mgmt`: utility functions to interact with S3 buckets
- `db_mgmt`: Implementation of the tables structure with SQLAlchemy
"""

from .utils import save_to_json
from .api_mgmt import get_coords, get_weather_forecast
from .scraping_mgmt import (
  scroll_down, scroll_to_bottom, load_more_results,
  scrape_hotel_info, scrape_hotel_urls, scrape_from_searchpage,
)
from .s3_mgmt import download_csv
from .db_mgmt import Base, Location, Hotel, WeatherIndicator, reflect_db

