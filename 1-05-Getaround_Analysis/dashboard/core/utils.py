# -*- coding: utf-8 -*-
"""
utility functions
"""

import re

import numpy as np




SUFFIXES = ["a", "f", "p", "n", "u", "m",
            "", "K", "M", "B", "T", "Qa", "Qu", "S", "Oc"] 
SCI_EXPONENTS = np.logspace(-18, 24, 15)




def numerize(x: float | int, decimals: int = 2)-> str:
    """
    Convert long numbers into human readable form.

    Parameters
    ----------
    x : float | int
        The number to convert.
    decimals : int, optional
        Number of decimal digits. The default is 2.
        
    Examples
    --------
    >>> numerize(1234567890)
    '1.23B'
    >>> numerize(0.000000123456, decimals=1)
    '123.5n'

    """
    sign = '-' if x < 0 else ''
    x = abs(x)
    i = np.searchsorted(SCI_EXPONENTS, x) - 1
    exponent, suffix = SCI_EXPONENTS[i], SUFFIXES[i]
    x_ = round(x / exponent, ndigits=decimals)
    return f'{sign}{x_}{suffix}'