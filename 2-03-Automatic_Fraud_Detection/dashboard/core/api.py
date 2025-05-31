# -*- coding: utf-8 -*-
"""

"""

import httpx



def probe_api(api_url: str)-> int:
    """
    Probe the API endpoint for availability.
    """
    r = httpx.get(f'{api_url}/test', follow_redirects=True)
    res = r.raise_for_status().json()
    return res


def get_pricing_models(api_url: str)-> dict[str, str]:
    """
    Get pricing models to appear in the selectbox.
    """
    r = httpx.get(f'{api_url}/pricing_models', follow_redirects=True,
                  timeout=3)
    return r.raise_for_status().json()


def get_categories(api_url: str)-> dict[str, list[str]]:
    """
    Get the categories for each categorical feature to appear in the
    corresponding selectbox.
    """
    r = httpx.get(f'{api_url}/categories', follow_redirects=True,
                  timeout=3)
    return r.raise_for_status().json()


def get_pricing(api_url: str,
                model: str,
                data: dict[str, bool | float | str])-> float:
    """
    Get the car pricing estimation.
    """
    r = httpx.post(f'{api_url}/predict/{model}',
                   follow_redirects=True, json=data, timeout=3)
    res = r.raise_for_status().json()
    return res['prediction']



