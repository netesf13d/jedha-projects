# -*- coding: utf-8 -*-
"""
Data management engine. Provides the following functionality:
- API calls to fetch both historical and forecast data from open-meteo. See
  https://open-meteo.com/en/docs
  https://open-meteo.com/en/docs/historical-weather-api
- Archive management in S3 buckets
- Data transfert to AWS RDS database
"""

from .utils import save_to_json
from .api_mgmt import get_coords, get_weather_forecast
from .scraping_mgmt import (
  scroll_down, scroll_to_bottom, load_more_results,
  scrape_hotel_info, scrape_hotel_urls, scrape_from_searchpage,
)
#from .s3_mgmt import *
#from .db_mgmt import *

