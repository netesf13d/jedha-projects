# -*- coding: utf-8 -*-
"""

"""

import sys
import time
from io import BytesIO
from typing import Any

import httpx
import numpy as np


INTERVALS = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600}

# =============================================================================
# Fetch from Kraken API https://docs.kraken.com/api/docs/rest-api/
# =============================================================================

def get_ticker(pair: str)-> dict[str, dict[str, float | int]]:
    """
    Get ticker information for the requested market `pair` (eg 'ETHUSD').

    Returns
    -------
    parsed_data : dict[str, np.ndarray]
        TODO DESCRIPTION.
        Today's prices start at midnight UTC.

    """
    params = {'pair': pair}
    r = httpx.get('https://api.kraken.com/0/public/Ticker', params=params,
                  follow_redirects=True)
    raw_data = r.raise_for_status().json()
    if raw_data['error']:
        raise ValueError(raw_data['error'], pair)
    data = raw_data['result']
    
    # Parse data
    data = next(iter(data.values()))
    parsed_data = {
        'ask': {'price': float(data['a'][0]),
                'volume': float(data['a'][2])},
        'bid': {'price': float(data['b'][0]),
                'volume': float(data['b'][2])},
        'last_trade': {'price': float(data['c'][0]),
                       'volume': float(data['c'][1])},
        'volume': {'today': float(data['v'][0]),
                   'last_24h': float(data['v'][1])},
        'vwap': {'today': float(data['p'][0]),
                 'last_24h': float(data['p'][1])},
        'trades': {'today': int(data['t'][0]),
                   'last_24h': int(data['t'][1])},
        'low': {'today': float(data['l'][0]),
                'last_24h': float(data['l'][1])},
        'high': {'today': float(data['h'][0]),
                 'last_24h': float(data['h'][1])},
        'open': float(data['o'])
    }
    return parsed_data


def get_ohlc(pair: str,
             interval: int = 1,
             since: float | None = None,
             )-> tuple[int, dict[str, np.ndarray]]:
    """
    Get OHLC data since the timestamp `since` (in seconds), with time frame
    `interval` for the selected `pair` (eg 'ETHUSD').
    
    The last entry in the OHLC array is for the current, not-yet-committed
    timeframe, and will always be present, regardless of the value of `since`.
    Returns up to 720 of the most recent entries (older data cannot be
    retrieved, regardless of the value of `since`).

    Returns
    -------
    last : int
        Timestamp of the last requested trade, in seconds.
        ID to be used as `since` when polling for new, commited OHLC data.
    parsed_data : dict[str, np.ndarray]
        TODO DESCRIPTION.

    """
    if interval not in INTERVALS:
        raise ValueError(f'`interval` must be in {INTERVALS}')
    params = {'pair': pair, 'interval': interval, 'since': since}
    r = httpx.get('https://api.kraken.com/0/public/OHLC', params=params,
                  follow_redirects=True)
    raw_data = r.raise_for_status().json()
    if raw_data['error']:
        raise ValueError(raw_data['error'], pair)
    data = raw_data['result']
    
    # Parse data
    last = int(data.pop('last'))
    data = np.array(next(iter(data.values())))
    parsed_data = {
        'time': data[:, 0].astype(np.float64),
        'open': data[:, 1].astype(np.float32),
        'high': data[:, 2].astype(np.float32),
        'low': data[:, 3].astype(np.float32),
        'close': data[:, 4].astype(np.float32),
        'vwap': data[:, 5].astype(np.float64),
        'volume': data[:, 6].astype(np.float64),
        'trades': data[:, 7].astype(np.float32)
    }
    return last, parsed_data


def get_trades(pair: str,
               since: float | int | None = None,
               count: int = 1000,
               )-> tuple[int, dict[str, np.ndarray]]:
    """
    Get `count` trades executed since the timestamp `since`, for the selected
    `pair` (eg 'ETHUSD').
    
    Notes
    -----
    - If `since` is None, return the last trades.
    - If `since` is an `int`, it represents the timestamp in nanoseconds
    - If `since` is a `float`, it represents the timestamp in seconds
    - `count` must be <= 1000

    Returns
    -------
    last : int
        Timestamp of the last requested trade, in nanoseconds.
        ID to be used as `since` when polling for subsequent trade data.
    parsed_data : dict[str, np.ndarray]
        TODO DESCRIPTION.

    """
    params = {'pair': pair, 'since': since, 'count': count}
    r = httpx.get('https://api.kraken.com/0/public/Trades', params=params,
                  follow_redirects=True)
    raw_data = r.raise_for_status().json()
    if raw_data['error']:
        raise ValueError(raw_data['error'])
    data = raw_data['result']
    
    # Parse data
    last = int(data.pop('last'))
    data = np.array(next(iter(data.values())))
    parsed_data = {
        'price': data[:, 0].astype(np.float64),
        'volume': data[:, 1].astype(np.float64),
        'time': data[:, 2].astype(np.float64),
        'is_buy': (data[:, 3] == 'b'), # otherwise it is a selling order
        'is_market': (data[:, 4] == 'm'), # otherwise it is a limit order
        'trade_id': data[:, 6].astype(int)
    }
    return last, parsed_data


def get_order_book(pair: str,
                   count: int = 100,
                   )-> dict[str, dict[str, np.ndarray]]:
    """
    Get `count` lowest asks / highest bids from the order book for the selected
    `pair` (eg 'ETHUSD').
    
    Notes
    -----
    - `count` must be <= 500

    Returns
    -------
    parsed_data : dict[str, np.ndarray]
        TODO DESCRIPTION.

    """
    params = {'pair': pair, 'count': count}
    r = httpx.get('https://api.kraken.com/0/public/Depth', params=params,
                  follow_redirects=True)
    raw_data = r.raise_for_status().json()
    if raw_data['error']:
        raise ValueError(raw_data['error'])
    data = raw_data['result']
    
    # Parse data
    data = next(iter(data.values()))
    asks = np.array(data['asks'], dtype=np.float64)
    bids = np.array(data['bids'], dtype=np.float64)
    parsed_data = {
        'asks': {'price': asks[:, 0],
                 'volume': asks[:, 1],
                 'time': asks[:, 2]},
        'bids': {'price': bids[:, 0],
                 'volume': bids[:, 1],
                 'time': bids[:, 2]},
    }
    return parsed_data


# =============================================================================
# Fetch from domestic API
# =============================================================================

CCP_API_URL = 'http://localhost:4000'

def probe_api():
    r = httpx.get(f'{CCP_API_URL}/hello', follow_redirects=True)
    raw_data = r.raise_for_status().text
    return int(raw_data)


def get_market_crashes(pair: str,
                       since: float = 0,
                       interval: int = 1440)-> dict[str, np.ndarray]:
    params = {'pair': pair, 'since': since, 'interval': interval}
    r = httpx.get(f'{CCP_API_URL}/crashes', params=params,
                  follow_redirects=True)
    raw_data = r.raise_for_status().content
    with np.load(BytesIO(raw_data), allow_pickle=False) as f:
        data = dict(f)
    return data


def get_crash_prob(pair: str,
                   since: float = 0,
                   interval: int = 1440)-> dict[str, np.ndarray]:
    params = {'pair': pair, 'since': since}
    r = httpx.get(f'{CCP_API_URL}/crash_probability', params=params,
                  follow_redirects=True)
    raw_data = r.raise_for_status().content
    with np.load(BytesIO(raw_data), allow_pickle=False) as f:
        data = dict(f)
    return data



# =============================================================================
# 
# =============================================================================


# data = get_ticker('XBTUSD')

# data = get_order_book('XBTUSD', 10)

# last, data = get_trades('XBTUSD', since=None, count=5)
# last2, data2 = get_trades('XBTUSD', since=last >> 1, count=5)

# last, data = get_ohlc('ETHUSD', interval=15) #, since=time.time() - 30*60)


# data = get_market_crashes('BTCUSD', 1706116133.1154096)

# data = get_crash_prob('BTCUSD', 1706116133.1154096, interval=1440)

# test = probe_api()


# addr = 'https://api.kraken.com/0/'
# tgt = 'public/OHLC'

# headers = {'Accept': 'application/json'}

# params = {'pair': 'XBTUSD',
#           'interval': 5,
#           'since': 0}

# r = httpx.get(f'{addr}{tgt}', params=params)
# data = r.json()

# tgt = 'test'
# params = "{'message': 'nice,bibi'}"
# params = {'n': 7} # {'pair': 'ETHBTC', 'since': 1735074259, 'interval': 5}
# r = requests.get(f'{addr}{tgt}', params=params)

# # print(r.text)

# with np.load(BytesIO(r.content), allow_pickle=False) as f:
#     data = dict(f)
# print(data)




