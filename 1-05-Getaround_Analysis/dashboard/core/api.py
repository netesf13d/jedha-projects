# -*- coding: utf-8 -*-
"""

"""

import sys
import time
from io import BytesIO
from typing import Any

import httpx



# =============================================================================
# 
# =============================================================================

CCP_API_URL = 'http://localhost:4000'

def probe_api():
    r = httpx.get(f'{CCP_API_URL}/hello', follow_redirects=True)
    raw_data = r.raise_for_status().text
    return int(raw_data)






