# -*- coding: utf-8 -*-
"""

"""

import time
from typing import Annotated, Literal

import httpx
import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..utils import iter_npz


rng = np.random.default_rng(1234)

INTERVALS = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600}

# =============================================================================
# 
# =============================================================================

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


def drop_idx(high: np.ndarray,
             low: np.ndarray,
                 nmax: int)-> np.ndarray:
    """
    

    Parameters
    ----------
    high : np.ndarray
        DESCRIPTION.
    low : np.ndarray
        DESCRIPTION.
    nmax : int
        DESCRIPTION.

    Returns
    -------
    idx: 2D np.ndarray of shape (*, 2)
        DESCRIPTION.

    """
    idx = []
    i0, i1 = 0, 0
    h0, lmin = -np.inf, np.inf
    i = 0
    while i < len(high):
        h, l = high[i], low[i]
        if h >= h0:
            if i0 < i1:
                idx.append((i0, i1))
            elif h0 > l:
                idx.append((i0, i))
            i0, i1 = i, i
            h0, lmin = h, l
        
        else:
            if l < lmin:
                i1, lmin = i, l
        
        if i - i1 >= nmax:
            idx.append((i0, i1))
            i = i1
            i0, i1 = i, i
            h0, lmin = h, l
        i += 1
    
    return np.array(idx)


def drop_events(time: np.ndarray,
                high: np.ndarray,
                low: np.ndarray,
                nmax: int,
                threshold: float = 0.1)-> dict[str, np.ndarray]:
    idx = drop_idx(high, low, nmax)
    drops = (high[idx[:, 0]] - low[idx[:, 1]]) / high[idx[:, 0]]
    events_idx = np.nonzero(drops >= threshold)
    return {'events': time[idx[events_idx]], 'variations': drops[events_idx]}



# =============================================================================
# 
# =============================================================================

router = APIRouter()


@router.get("/ohlcvt", tags=['history'])
async def ohlcvt(
    pair: str,
    since: float,
    interval: int = 1,
    )-> dict[str, bytes]:
    """
    Fetch OHLCVT data from storage.

    Parameters
    ----------
    pair : str
        Asset pair to get data for.
        Example: 'XBTUSD' for bitcoin vs US dollar.
    
    since : float
        The timestamp, in seconds, since which data is returned.
    
    interval : int, optional
        The timeframe interval in minutes. The default is 1.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    n = int((time.time() - since) / (interval * 60))
    ohlcvt_data = {
        'time': np.linspace(since, since+n*60*interval, n, endpoint=False),
        'open': rng.random(size=n, dtype=np.float32),
        'high': rng.random(size=n, dtype=np.float32),
        'low': rng.random(size=n, dtype=np.float32),
        'close': rng.random(size=n, dtype=np.float32),
        'volume': rng.random(size=n, dtype=np.float32),
        'trades': rng.random(size=n, dtype=np.float64),
    }
    gen_ = iter_npz(ohlcvt_data)
    
    return StreamingResponse(gen_, media_type='zip-archive/npz')



@router.get("/crashes", tags=["Crash detection"])
async def crashes(
    pair: str,
    since: float | int = 0,
    interval: int = 1440,
    )-> dict[str, bytes]:
    """
    Fetch OHLCVT data from storage.

    Parameters
    ----------
    pair : str
        Asset pair to get data for.
        Example: 'XBTUSD' for bitcoin vs US dollar.
    since : float
        The timestamp, in seconds, since which data is returned.
    interval : int, optional
        The timeframe interval in minutes. The default is 1.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    nmax = np.ceil(1440 / interval).astype(int) # 24h
    _, ohlc = get_ohlc(pair, interval, since)
    crash_data = drop_events(ohlc['time'], ohlc['high'], ohlc['low'],
                              nmax, 0.1)

    gen_ = iter_npz(crash_data)
    
    return StreamingResponse(gen_, media_type='zip-archive/npz')


