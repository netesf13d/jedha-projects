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

def probe_api(api_url: str):
    r = httpx.get(f'{api_url}/test', follow_redirects=True)
    res = r.raise_for_status().json()
    return res


def get_models(api_rul:str):
    pass


def get_categories(api_url: str):
    pass


def get_pricing(api_url: str,
                data: dict[str, bool | float | str],
                model: str)-> float:
    r = httpx.post(f'{api_url}/predict', follow_redirects=True, data=data)
    res = r.raise_for_status().json()
    return res



