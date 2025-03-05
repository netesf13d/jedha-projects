#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import json


init_config = {
    'display_crashes': False,
    
    }

with open('./init_session_state.json', 'wt', encoding='utf-8') as f:
    json.dump(init_config, f, indent=2)

with open('../data/ohlcvt_data_index.json', 'rt', encoding='utf-8') as f:
    crypto_index = json.load(f)
crypto_codes = crypto_index['codes']
assets = crypto_index['assets']
asset_pairs = {f'{a1}/{a2}': [a1, a2]
               for a1, a2_list in assets.items()
               for a2 in a2_list}
markets = list(asset_pairs)


