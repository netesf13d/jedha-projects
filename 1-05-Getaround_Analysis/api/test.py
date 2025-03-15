# -*- coding: utf-8 -*-
"""

"""

import sys

import requests


# =============================================================================
# 
# =============================================================================

def probe_api(url: str = 'http://localhost:4000/')-> int:
    """
    Probe the API with the test endpoint.
    """
    r = requests.get(f'{addr}test')
    r.raise_for_status()
    if (res:=r.json()) != 1:
        raise ValueError(f'invalid value: {res}')
    return 1
    

def test_predict(data: list, model: str, url: str = 'http://localhost:4000/'):
    r = requests.post(f'{addr}predict', data='test')
    r.raise_for_status()
    return r.json()
    if (res:=r.json()) != 1:
        raise ValueError(f'invalid value: {res}')
    return 1



if __name__ == '__main__':
    addr = 'http://localhost:4000/'
    
    print('Probe API:', bool(probe_api(addr)))

    data = ['Audi', 132979, 112, 'diesel', 'brown', 'estate',
            True, True, False, False, True, True, True, 117]
    res = test_predict(data, 'ridge', addr)
    

