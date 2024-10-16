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
import scrapy



# =============================================================================
# 
# =============================================================================

coords = (-0.7024738, 49.2764624)
coords = {'lon': -0.7024738, 'lat': 49.2764624}
display_name = ""

params = {
    'ss': display_name,
    'checkin': '2024-10-24',
    'checkout': '2024-11-01',
    'group_adults': 2,
    'group_children': 1,
    'no_rooms': 1}



class BookingSpider(scrapy.Spider):
    name = "hotels_info"
    
    def start_requests(self):
        pass
        
    
    def parse():
        pass


url = """
https://www.booking.com/searchresults.en-gb.html?ss=94250%2C+France&ssne=Paris&ssne_untouched=Paris&label=gen173nr-1BCAEoggI46AdIM1gEaE2IAQGYAQm4ARjIAQ_YAQHoAQGIAgGoAgS4AoCAtLgGwAIB0gIkMzJiOGJhZTctNDk2Yi00ZTk5LTg2MWQtYzFjYmJkOTMyZjFk2AIF4AIB&sid=438db7b646a86ed2276c339865eec369&aid=304142&lang=en-gb&sb=1&src_elem=sb&src=searchresults&dest_id=9608074&dest_type=hotel&ac_position=0&ac_click_type=b&ac_langcode=en&ac_suggestion_list_length=1&search_selected=true&search_pageview_id=53ad50f3c34f08d7&ac_meta=GhA1M2FkNTBmM2MzNGYwOGQ3IAAoATICZW46DTk0MjUwLCBGcmFuY2VAAEoAUAA%3D&checkin=2024-10-18&checkout=2024-10-25&group_adults=2&no_rooms=1&group_children=0#map_opened
"""

url2 = """
https://www.booking.com/searchresults.en-gb.html?src=searchresults&dest_type=landmark&latitude=48.2494853&longitude=7.3444831&ac_position=0&ac_click_type=g&ac_langcode=en-gb&ac_suggestion_list_length=100&search_selected=true&checkin=2024-10-18&checkout=2024-10-25&group_adults=2&no_rooms=1&group_children=0#map_opened
"""

