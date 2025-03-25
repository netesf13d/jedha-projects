# -*- coding: utf-8 -*-
"""

"""

import httpx



def probe_api(api_url: str)-> int:
    """
    TODO doc
    """
    r = httpx.get(f'{api_url}/test', follow_redirects=True, timeout=3)
    res = r.raise_for_status().json()
    return res


def get_pricing_models(api_url: str)-> dict[str, str]:
    """
    TODO doc
    """
    r = httpx.get(f'{api_url}/pricing_models', follow_redirects=True,
                  timeout=5)
    return r.raise_for_status().json()


def get_categories(api_url: str)-> dict[str, list[str]]:
    """
    TODO doc
    """
    r = httpx.get(f'{api_url}/categories', follow_redirects=True,
                  timeout=5)
    return r.raise_for_status().json()


def get_pricing(api_url: str,
                data: dict[str, bool | float | str],
                model: str)-> float:
    """
    TODO
    """
    r = httpx.post(f'{api_url}/predict', follow_redirects=True, data=data)
    res = r.raise_for_status().json()
    return res



