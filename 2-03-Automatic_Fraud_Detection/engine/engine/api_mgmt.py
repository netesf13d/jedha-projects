# -*- coding: utf-8 -*-
"""

"""

import json

import httpx



def get_root(api_url: str)-> dict[str, dict[str, float | int]]:
    """
    Probe the API endpoint. Return the contents of the main page.
    """
    r = httpx.get(api_url, follow_redirects=True)
    raw_data = r.raise_for_status().text
    return raw_data


def get_transaction(api_url: str)-> dict[str, list]:
    """
    Get a sample transaction data.
    !!!
    """
    r = httpx.get(api_url, follow_redirects=True,
                  timeout=httpx.Timeout(10, read=None))
    raw_data = r.raise_for_status().text
    return json.loads(json.loads(raw_data))



def detect_fraud(api_url: str,
                 model: str,
                 data: dict[str, bool | float | str])-> int:
    """
    !!!
    Get the car pricing estimation.
    """
    r = httpx.post(f'{api_url.strip("/")}/predict/{model}',
                   follow_redirects=True, json=data, timeout=3)
    res = r.raise_for_status().json()
    return res['prediction']