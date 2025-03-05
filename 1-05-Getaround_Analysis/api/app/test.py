# -*- coding: utf-8 -*-
"""

"""

import sys
from io import BytesIO
from typing import Any

import requests
import numpy as np


# =============================================================================
# 
# =============================================================================

def test(addr: str = 'http://localhost:4000/',
         tgt: str = 'test',
         params: dict | None = None)-> Any:
    params = {} if params is None else params
    r = requests.get(f'{addr}{tgt}', params=params)
    print(r)
    if r.status_code == 200:
        with np.load(BytesIO(r.content), allow_pickle=False) as f:
            data = dict(f)
        return data
    else:
        return r


addr = 'http://localhost:4000/'
tgt = 'custom_greetings'
params = {'name': 'bibi'}

# r = requests.get(f'{addr}{tgt}', params=params)

# tgt = 'test'
# params = "{'message': 'nice,bibi'}"
# params = {'n': 7} # {'pair': 'ETHBTC', 'since': 1735074259, 'interval': 5}
# r = requests.get(f'{addr}{tgt}', params=params)

# # print(r.text)

# with np.load(BytesIO(r.content), allow_pickle=False) as f:
#     data = dict(f)
# print(data)


ohlcvt_params = {'pair': 'ETHUSD', 'since': 1735074259.1, 'interval': 10}
t1 = test(addr, 'ohlcvt', params=ohlcvt_params)

crash_params = {'pair': 'ETHUSD', 'since': 173507425.1}
t2 = test(addr, 'crashes', params=crash_params)

crash_prob_params = {'pair': 'ETHUSD', 'since': 1735074259.1, 'interval': 10}
t3 = test(addr, 'ohlcvt', params=crash_prob_params)


