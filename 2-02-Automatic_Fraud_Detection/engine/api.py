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



# =============================================================================
# Fetch from Jedha API
# =============================================================================

def get_root()-> dict[str, dict[str, float | int]]:
    """
    
    """
    r = httpx.get('https://real-time-payments-api.herokuapp.com/',
                  follow_redirects=True)
    raw_data = r.raise_for_status().text
    return raw_data



def get_transaction()-> dict[str, list]:
    """
    
    """
    r = httpx.get('https://real-time-payments-api.herokuapp.com/current-transactions',
                  follow_redirects=True)
    raw_data = r.raise_for_status().text
    return json.loads(raw_data)



