# -*- coding: utf-8 -*-
"""

"""

import json
# from datetime import datetime, UTC

import httpx
import numpy as np



def get_root(api_url: str)-> dict[str, dict[str, float | int]]:
    """
    Probe the transaction API endpoint. Return the contents of the main page.
    """
    r = httpx.get(api_url, follow_redirects=True)
    raw_data = r.raise_for_status().text
    return raw_data


def get_transaction(api_url: str)-> tuple[str, dict[str, list]]:
    """
    Get a sample transaction data.
    
    >>> api_url = 'http://localhost:8000'
    >>> get_transaction(api_url)
    {'columns': ['cc_num', 'merchant', 'category', 'amt', 'first', 'last',
                 'gender', 'street', 'city', 'state', 'zip', 'lat', 'long',
                 'city_pop', 'job', 'dob', 'trans_num', 'merch_lat',
                 'merch_long', 'is_fraud', 'current_time'],
     'index': [36984],
     'data': [[6011652924285713, 'fraud_Kerluke Inc', 'misc_net', 2.8,
               'Kathryn', 'Smith', 'F', '19838 Tonya Prairie Apt. 947',
               'Rocky Mount', 'MO', 65072, 38.2911, -92.7059, 1847,
               'Tax inspector', '1988-10-26', 'c1ff96799d79e1434b6dc33718ecd48c',
               38.97795, -92.409559, 0,1748545063909]]}
    """
    r = httpx.get(f'{api_url.strip("/")}/current-transactions',
                  follow_redirects=True,
                  timeout=httpx.Timeout(10, read=None))
    raw_data = r.raise_for_status().text
    # return (datetime.now(tz=UTC).isoformat(),
    return json.loads(json.loads(raw_data))



def detect_fraud(api_url: str,
                 model: str,
                 data: dict[str, bool | float | str])-> int:
    """
    !!!
    Get the car pricing estimation.
    """
    ## remove numpy dtypes in data
    for k, v in data.items():
        if isinstance(v, np.float64):
            data[k] = float(v)
        elif isinstance(v, np.int64):
            data[k] = int(v)
    print(data)
    r = httpx.post(f'{api_url.strip("/")}/predict/{model}',
                   follow_redirects=True, json=data, timeout=3)
    res = r.raise_for_status().json()
    return res['prediction']