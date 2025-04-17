# -*- coding: utf-8 -*-
"""
Utilities
"""

import json
import os

def save_to_json(fname: str, data: dict):
    """
    Save scraped data to json format.
    """
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wt', encoding='utf-8') as f:
        json.dump(data, f, indent=2)