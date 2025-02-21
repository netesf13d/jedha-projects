# -*- coding: utf-8 -*-
"""

"""

import sys
import os
import re
import unicodedata
import argparse
import time
import json
import csv
from pathlib import Path


# from bs4 import BeautifulSoup as bs
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
# from selenium.webdriver.chrome.options import Options


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('disable-notifications')
webd_path = '../chromedriver_118.exe'


tgt_xpaths = {
    # properties
    'property-card': '//div[@data-testid="property-card" and @role="listitem"]',
    'title': './/div[@data-testid="title"]',
    
    'link': './/a[@data-testid="title-link"]',
    
    'rating': './/div[@class="a3b8729ab1 d86cee9b25"]',
    'rating_comment': './/div[@class="a3b8729ab1 e6208ee469 cb2cbb3ccb"]',
    'georating': './/div[@class="a3332d346a"]'
    }


# =============================================================================
# 
# =============================================================================

def scrape_properties(driver, urls: list[str])-> dict:
    
    for url in urls:
        driver.get(url)
        properties = driver.find_elements(By.XPATH, tgt_xpaths['property-card'])
        for p in properties:
            pass


def get_descriptions(driver, urls: list[str])-> dict:
    for url in urls:
        driver.get(url)


def try_find_element(web_element, xpath: str)-> str | None:
    
    try:
        web_element.find_element(By.XPATH, xpath).text
    except NoSuchElementException:
        return None


# =============================================================================
# 
# =============================================================================

coords = {'latitude': 49.4404591, 'longitude': 1.0939658} # Rouen
url = ('https://www.booking.com/searchresults.en-gb.html?'
       f'latitude={coords["latitude"]}&longitude={coords["longitude"]}')



driver = webdriver.Chrome(options=chrome_options)
driver.get(url)

propertycard_xpath = '//div[@data-testid="property-card" and @role="listitem"]'
title_xpath = './/div[@data-testid="title"]'



properties = driver.find_elements(By.XPATH, tgt_xpaths['property-card'])
for p in properties:
    
    name = p.find_element(By.XPATH, tgt_xpaths['title']).text
    
    url = p.find_element(By.XPATH, tgt_xpaths['link']).get_attribute('href')
    
    rating = try_find_element(p, tgt_xpaths['rating'])
    rating_comment = try_find_element(p, tgt_xpaths['rating_comment'])
    georating = try_find_element(p, tgt_xpaths['georating'])
    
    break



driver.Dispose()


# =============================================================================
# Connection and webdriver control
# =============================================================================

class Facebook_Session():
    address = "https://www.facebook.com"
    
    def __init__(self, webd_exec_path: str):
        ...
        


def start_session():
    ...
    

def apply_cookie_policy(driver, policy: str = 'only_necessary')-> None:
    if policy == 'only_necessary':
        xp_val = "//button[@data-cookiebanner='accept_only_essential_button']"
    if policy == 'accept_all':
        xp_val = "//button[@data-cookiebanner='accept_button']"
    if not xp_val:
        raise ValueError("policy must be 'only_necessary' or 'accept_all'")
    driver.find_element(by=By.XPATH, value=xp_val).click()


def login(driver, email: str, passwd: str):
    # Enter credentials
    driver.find_element(by=By.NAME, value="email").send_keys(email)
    driver.find_element(by=By.NAME, value="pass").send_keys(passwd)
    # Connection
    driver.find_element(by=By.NAME, value="login").click()


# =============================================================================
# 
# =============================================================================

def get_pages(driver,
              search: list[str] = None,
              friends: list[str] = None,
              urls: list[str] = None)-> list[str]:
    pages = []
    if search is not None:
        pages += get_search(search)
    if friends is not None:
        pages += get_friends(friends)
    if urls is not None:
        pages += get_urls(urls)
    return pages
    
    

def get_search(driver, search: list[str])-> list[str]:
    ...


def get_friends(driver, friends: list[str])-> list[str]:
    ...


def get_urls(driver, urls: list[str])-> list[str]:
    address = "https://www.facebook.com"
    # build full fleged facebook urls
    std_urls = []
    for url in urls:
        if not url:
            print("ignoring empty url...")
            continue
        url = url.split('/')
        page = url[-1] if url[-1] else url[-2]
        std_urls.append(address + '/' + page)
    # check that the pages exist
    pages = []
    for page in std_urls:
        if check_available(page):
            pages.append(page)
    return pages


def check_available(url: str):
    # get element ".//img[@class='x1b0d499']" which indicates page innaccessible
    ...


# =============================================================================
# 
# =============================================================================






