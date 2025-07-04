# -*- coding: utf-8 -*-
"""

"""

import json

import httpx



def probe_transaction_api(api_url: str)-> str:
    """
    Probe the transaction API endpoint. Return the contents of the main page.
    """
    r = httpx.get(api_url, follow_redirects=True,
                  timeout=httpx.Timeout(3, read=None))
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
    return json.loads(json.loads(raw_data))


def probe_fraud_detection_api(api_url: str)-> dict[str, int]:
    """
    Probe the fraud detection API endpoint.
    """
    r = httpx.get(f'{api_url.strip("/")}/test', follow_redirects=True,
                  timeout=httpx.Timeout(3, read=None))
    raw_data = r.raise_for_status().text
    return raw_data


def detect_fraud(api_url: str,
                 model: str,
                 data: dict[str, bool | float | str])-> int:
    """
    !!!
    Get the car pricing estimation.
    """
    ## remove numpy dtypes in data
    # for k, v in data.items():
    #     data[k] = getattr(v, 'tolist', lambda: v)()
    r = httpx.post(f'{api_url.strip("/")}/predict/{model}',
                   follow_redirects=True, json=data, timeout=3)
    res = r.raise_for_status().json()
    return res['prediction']