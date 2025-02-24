# -*- coding: utf-8 -*-
"""
Convenience functions for web-scaping of booking.com with Selenium.
"""

import time

from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException



# xpaths for properties elements
searchpage_xpaths = {
    'property-card': '//div[@data-testid="property-card" and @role="listitem"]',
    
    'link': './/a[@data-testid="title-link"]',
    'title': './/div[@data-testid="title"]',
    'description': './/div[@class="abf093bdfe"]',
    'rating': './/div[@class="a3b8729ab1 d86cee9b25"]',
    'georating': './/span[@class="a3332d346a"]',
    }

# xpath for hotel pages elements
hotel_xpaths = {
    'hotelchars': '//div[@class="hotelchars"]',
    
    'hotel_id': '//input[@type="hidden" and @name="hotel_id"]',
    'title': './/h2[@class="d2fee87262 pp-header__title"]',
    'description': '//p[@data-testid="property-description"]',
    'rating': './/div[@class="a3b8729ab1 d86cee9b25"]',
    'georating': './/span[@class="review-score-badge"]',
    }

# =============================================================================
# Functions
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
    rating = rating.replace(',', '.')
    
    georating = _try_find_element(hotel, hotel_xpaths['georating'])
    georating = georating.replace(',', '.')
    
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
        if rating is not None:
            rating = rating.rsplit('\n', 1)[-1].replace(',', '.')
        
        georating = _try_find_element(hotel, searchpage_xpaths['georating'])
        if georating is not None:
            georating = georating.rsplit(' ', 1)[-1].replace(',', '.')
        
        hotel_infos.append({'url': url,
                            'name': name,
                            'description': description,
                            'rating': rating,
                            'georating': georating})
    
    return hotel_infos


# =============================================================================
# The script below scrapes each individual hotel page. This takes a lot of time
# and the website is likely to catch selenium as a robot and block it.
# =============================================================================

# import csv
# from selenium import webdriver
# from selenium.common.exceptions import StaleElementReferenceException
# from utils import save_to_json

# ## setup driver with options to prevent detection
# ## https://stackoverflow.com/questions/53039551/selenium-webdriver-modifying-navigator-webdriver-flag-to-prevent-selenium-detec/53040904#53040904
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


# with open("./data/locations.csv", 'rt', encoding='utf-8') as f:
#     reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_STRINGS)
#     next(reader, None) # remove header
#     locations = [row for row in reader]

# search_urls = {int(loc[0]): ('https://www.booking.com/searchresults.en-gb.html?'
#                             f'latitude={loc[3]}&longitude={loc[4]}')
#                for loc in locations}

# ## open 2 tabs
# # search tab
# driver.get('https://www.booking.com')
# search_window = driver.current_window_handle
# # hotel description tab
# driver.switch_to.new_window('tab')
# driver.get('https://www.booking.com')
# hotel_window = driver.current_window_handle

# for i, search_url in search_urls.items():
#     # get urls of hotel pages
#     driver.switch_to.window(search_window)
#     hotel_urls = scrape_hotel_urls(driver, search_url) # scrape on search
#     hotel_urls = hotel_urls[:30]
    
#     # scrape hotel pages
#     driver.switch_to.window(hotel_window)
#     hotel_infos = []
#     for url in hotel_urls:
#         time.sleep(0.5)
#         try:
#             hotel_infos.append(scrape_hotel_info(driver, url))
#         except StaleElementReferenceException:
#             time.sleep(0.5)
#             hotel_infos.append(scrape_hotel_info(driver, url))

#     # save intermediate results to json
#     save_to_json(f'../data/temp/hotels/{i}.json', hotel_infos)

# driver.quit()



