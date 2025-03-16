# -*- coding: utf-8 -*-
"""

"""

import io

import numpy as np


# =============================================================================
# 
# =============================================================================

def iter_npz(data: dict[str, np.ndarray])-> bytes:
    """
    TODO docs

    Parameters
    ----------
    data : dict[str, np.ndarray]
        DESCRIPTION.

    Yields
    ------
    bytes
        DESCRIPTION.

    """
    with io.BytesIO() as f:
        np.savez_compressed(f, **data)
        f.seek(0)
        yield from f



