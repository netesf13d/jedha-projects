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


def ob_template(path: str, depth: int, data_fmt: dict)-> tuple[str, str]:
    """
    TODO doc

    Parameters
    ----------
    depth : int
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    str
        DESCRIPTION.

    """
    with open(path, 'rt', encoding='utf-8') as f:
        template = f.read()
    
    sstyle = re.search('</style>', template)
    i = 0 if not sstyle else sstyle.span()[1] + 1
    style = template[:i].replace('{depth}', str(depth))
    template = template[i:]

    if not (sa:=re.search(' +<tr class="ask">', template)):
        raise ValueError("'ask' row template not found")
    if not (sb:=re.search(' +<tr class="bid">', template)):
        raise ValueError("'bid' row template not found")
    
    head = template[:sa.span()[0]]
    # head = head.replace('{', '{{').replace('}', '}}')
    ask = template[sa.span()[0]:sb.span()[0]].format(**data_fmt)
    bid = template[sb.span()[0]:].format(**data_fmt)
    i = re.search(' +</tr>', bid).span()[1]+1
    bid, tail = bid[:i], bid[i:]
    
    return style, head + depth*ask + depth*bid + tail




# def _build_order_book_table():
#     import pandas as pd
#     from pandas.io.formats.style import Styler
    
#     styles = [
#         {'selector': 'table', 'props': 'width: 80%;'},
#         {'selector': 'th, td', 'props': 'padding-left: 10px; padding-right: 20px; text-align: right;'},
#         {'selector': '.row{depth}', 'props': 'border-top: 1px solid silver;'},
#         {'selector': 'th', 'props': 'font-weight: normal;'},
#         {'selector': 'tbody', 'props': 'font-weight: bold;'},
#         {'selector': '.ask_price', 'props': 'color: red;'},
#         {'selector': '.bid_price', 'props': 'color: green;'},
#         ]
    
#     df = pd.DataFrame(
#         [['{{}}', '{{}}', '{{}}'], ['{{}}', '{{}}', '{{}}']],
#         columns=['PRICE', 'QUANTITY ({asset1})', 'VOLUME ({asset2})'])
#     classes = [['ask_price', 'ask_qty1', 'ask_qty2'],
#                ['bid_price', 'bid_qty1', 'bid_qty2']]
#     classes_df = pd.DataFrame(classes, columns=df.columns)
    
#     styler = Styler(df, uuid='ob', cell_ids=False)
#     styler = styler.set_td_classes(classes_df) \
#                    .hide(axis='index') \
#                    .set_table_styles(styles)
    
#     return styler.to_html()
