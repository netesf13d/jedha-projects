# -*- coding: utf-8 -*-
"""

"""

import time
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..utils import iter_npz


rng = np.random.default_rng(1234)


# =============================================================================
# 
# =============================================================================








# =============================================================================
# 
# =============================================================================

router = APIRouter()


@router.get("/crash_probability", tags=["Crash prediction"])
async def crash_probability(
    pair: str,
    since: float | int = 0,
    interval: int = 1
    )-> dict[str, bytes]:
    """
    Fetch OHLCVT data from storage.

    Parameters
    ----------
    pair : str
        Asset pair to get data for.
        Example: 'XBTUSD' for bitcoin vs US dollar.
    since : int
        The timestamp, in seconds, since which data is returned.
    interval : int, optional
        The timeframe interval in minutes. The default is 1.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    since = max(1706116133, since)
    n = int((time.time() - since) / (interval * 60))
    
    times = np.linspace(since, since+n*interval*60, n, endpoint=False)
    pcrash = np.clip(np.cumsum(rng.normal(0, 0.01, n)), 0, 1)
    
    crash_prob = {
        'time': times,
        'prob': pcrash,
    }
    gen_ = iter_npz(crash_prob)
    
    return StreamingResponse(gen_, media_type='zip-archive/npz')

