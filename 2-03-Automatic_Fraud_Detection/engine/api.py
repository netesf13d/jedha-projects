# -*- coding: utf-8 -*-
"""

"""

import sys
import json
import time
from io import BytesIO
from typing import Any


import httpx
import numpy as np

API_URL = 'https://charlestng-real-time-fraud-detection.hf.space/'

# =============================================================================
# Fetch from Jedha API
# =============================================================================

def get_root()-> dict[str, dict[str, float | int]]:
    """
    !!!
    """
    r = httpx.get(API_URL, follow_redirects=True)
    raw_data = r.raise_for_status().text
    return raw_data



def get_transaction()-> dict[str, list]:
    """
    !!!
    """
    url = API_URL + 'current-transactions'
    r = httpx.get(url, follow_redirects=True,
                  timeout=httpx.Timeout(10, read=None))
    raw_data = r.raise_for_status().text
    return json.loads(json.loads(raw_data))




a = get_transaction()
