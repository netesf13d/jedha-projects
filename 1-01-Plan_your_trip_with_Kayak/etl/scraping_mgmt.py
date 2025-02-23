# -*- coding: utf-8 -*-
"""

"""

# import json
# import os
import time

# from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

# from .utils import save_to_json


# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('disable-notifications')
# webd_path = '../chromedriver_118.exe'


# xpaths for properties elements
searchpage_xpaths = {
    'property-card': '//div[@data-testid="property-card" and @role="listitem"]',
    
    'link': './/a[@data-testid="title-link"]',
    'title': './/div[@data-testid="title"]',
    'description': './/div[@class="abf093bdfe"]',
    'rating': './/div[@class="a3b8729ab1 d86cee9b25"]',
    'georating': './/span[@class="a3332d346a"]',
    }


hotel_xpaths = {
    'hotelchars': '//div[@class="hotelchars"]',
    
    'hotel_id': '//input[@type="hidden" and @name="hotel_id"]',
    'title': './/h2[@class="d2fee87262 pp-header__title"]',
    'description': '//p[@data-testid="property-description"]',
    'rating': './/div[@class="a3b8729ab1 d86cee9b25"]',
    'georating': './/span[@class="review-score-badge"]',
    }

# =============================================================================
# 
# =============================================================================

def scroll_down(driver, scroll_pause: float = 0.4)-> bool:
    """
    Scroll down through the web page.

    Returns
    -------
    enf_of_page_flag : bool
        True if the end of page was reached by scrolling.
    """
    # Get scroll height
    cur_height = driver.execute_script("return document.body.scrollHeight")
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # Wait to load page
    time.sleep(scroll_pause)
    
    new_height = driver.execute_script("return document.body.scrollHeight")
    
    # Send end of page flag
    return new_height == cur_height


def scroll_to_bottom(driver, scroll_pause: float = 0.4):
    """
    Scroll to the bottom of the current page.
    """
    end_page = False
    while not end_page:
        end_page = scroll_down(driver, scroll_pause)


def load_more_results(driver):
    """
    Click on the 'laod more results' button at the end of the page.
    """
    load_button = ('//button[@class="a83ed08757 c21c56c305 bf0537ecb5 '
                   'f671049264 af7297d90d c0e0affd09"]')
    driver.find_element(By.XPATH, load_button).click()


def _try_find_element(web_element, xpath: str)-> str | None:
    """
    Try to find the text content of an element for its xpath, return None if
    not found.
    """
    try:
        return web_element.find_element(By.XPATH, xpath).text
    except NoSuchElementException:
        # print(f'failed to get {xpath}')
        return None


def scrape_hotel_info(driver, hotel_url: str)-> dict[str, str]:
    """
    Get various hotels informations from the hotel description page element.
    """
    # load hotel page
    driver.get(hotel_url)
    hotel = driver.find_element(By.XPATH, hotel_xpaths['hotelchars'])
    
    # booking.com hotel identifier
    hotel_id = driver.find_element(By.XPATH, hotel_xpaths['hotel_id'])
    hotel_id = hotel_id.get_attribute('value')
    
    name = hotel.find_element(By.XPATH, hotel_xpaths['title']).text
    
    description = _try_find_element(hotel, hotel_xpaths['description'])
    
    rating = _try_find_element(hotel, hotel_xpaths['rating'])
    rating = None if rating is None else rating.rsplit(' ', 1)[-1]
    
    georating = _try_find_element(hotel, hotel_xpaths['georating'])
    
    hotel_info = {'hotel_id': hotel_id,
                  'url': hotel_url,
                  'name': name,
                  'description': description,
                  'rating': rating,
                  'georating': georating,
                  }
    return hotel_info


def scrape_hotel_urls(driver, search_url: str)-> dict:
    """
    Scrape a list of hotels urls from the search results page.
    The browser must have loaded the results page.
    """
    # load the whole search page contents by scolling down
    # booking.com does not implement infinite scoll hence no infinite loop
    driver.get(search_url)
    scroll_to_bottom(driver)
    
    hotels = driver.find_elements(By.XPATH, searchpage_xpaths['property-card'])
    
    hotel_urls = []
    for h in hotels:
        url = h.find_element(By.XPATH, searchpage_xpaths['link'])
        hotel_urls.append(url.get_attribute('href').split('?')[0])
    return hotel_urls


def scrape_from_searchpage(driver,
                           search_url: str,
                           limit: int = 30)-> list[dict[str, str]]:
    """
    Get various hotels informations from the hotel description page element.
    """
    # load the whole search page contents by scolling down
    # booking.com does not implement infinite scoll hence no infinite loop
    driver.get(search_url)
    scroll_to_bottom(driver)
    
    hotels = driver.find_elements(By.XPATH, searchpage_xpaths['property-card'])
    
    hotel_infos = []
    for hotel in hotels[:limit]:
        url = hotel.find_element(By.XPATH, searchpage_xpaths['link'])
        url = url.get_attribute('href').split('?')[0]
        
        name = hotel.find_element(By.XPATH, searchpage_xpaths['title']).text
        
        description = _try_find_element(hotel, searchpage_xpaths['description'])
        
        rating = _try_find_element(hotel, searchpage_xpaths['rating'])
        rating = None if rating is None else rating.rsplit('\n', 1)[-1]
        
        georating = _try_find_element(hotel, searchpage_xpaths['georating'])
        georating = None if georating is None else georating.rsplit(' ', 1)[-1]
        
        hotel_infos.append({'url': url,
                            'name': name,
                            'description': description,
                            'rating': rating,
                            'georating': georating})
    
    return hotel_infos


# =============================================================================
# 
# =============================================================================
# from selenium import webdriver

# coords = [{'latitude': 49.4404591, 'longitude': 1.0939658}, # Rouen
#           {'latitude': 49.2764624, 'longitude': -0.7024738}]

# search_urls = [('https://www.booking.com/searchresults.en-gb.html?'
#                 f'latitude={coord["latitude"]}&longitude={coord["longitude"]}')
#                for coord in coords]


# options = webdriver.ChromeOptions()
# options.add_argument("--start-maximized")
# options.add_argument("--disable-blink-features=AutomationControlled")
# options.add_experimental_option("excludeSwitches", ["enable-automation"])
# options.add_experimental_option('useAutomationExtension', False)
# driver = webdriver.Chrome(options=options)
# driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
# driver.execute_cdp_cmd(
#     'Network.setUserAgentOverride',
#     {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'}
# )
# driver.implicitly_wait(1)


# for search_url in search_urls[:2]:
#     hotel_infos = scrape_from_searchpage(driver, search_url, limit=5) # scrape on search

    # driver.get(search_url)
    # scroll_to_bottom(driver)
    
    # hotels = driver.find_elements(By.XPATH, searchpage_xpaths['property-card'])
    
    # hotel_infos = {}
    # hotel = hotels[0]
    # url = hotel.find_element(By.XPATH, searchpage_xpaths['link'])
    # url = url.get_attribute('href').split('?')[0]
    
    # name = hotel.find_element(By.XPATH, searchpage_xpaths['title']).text
    
    # description = _try_find_element(hotel, searchpage_xpaths['description'])
    
    # rating = _try_find_element(hotel, searchpage_xpaths['rating'])
    # rating = None if rating is None else rating.rsplit(' ', 1)[-1]
    
    # georating = _try_find_element(hotel, searchpage_xpaths['georating'])
    
    # save_to_json(f'./data/temp/{hex(hash(search_url))[-16:]}.json',
    #              hotel_infos)

