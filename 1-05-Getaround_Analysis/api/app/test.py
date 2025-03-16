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
    r = requests.get(f'{url}test')
    r.raise_for_status()
    if (res:=r.json()) != 1:
        raise ValueError(f'invalid value: {res}')
    return 1
    

def test_get_car_models(url: str = 'http://localhost:4000/')-> int:
    """
    
    """
    r = requests.get(f'{url}car_models')
    r.raise_for_status()
    if (res:=r.json()) != 1:
        raise ValueError(f'invalid value: {res}')
    return 1


def test_predict(data: list, model: str, url: str = 'http://localhost:4000/'):
    r = requests.post(f'{url}predict', data={'model': 'test'})
    r.raise_for_status()
    return r.json()
    if (res:=r.json()) != 1:
        raise ValueError(f'invalid value: {res}')
    return 1


d = {'model_key': 'Audi',
     'mileage': 132979,
     'engine_power': 112,
     'fuel': 'diesel',
     'paint_color': 'brown',
     'car_type': 'estate',
     'private_parking_available': True,
     'has_gps': True,
     'has_air_conditioning': False,
     'automatic_car': False,
     'has_getaround_connect': True,
     'has_speed_regulator': True,
     'winter_tires': True}
r = requests.post("http://localhost:4000/predict", json={'data': d})



# if __name__ == '__main__':
#     addr = 'http://localhost:4000/'
    
#     print('Probe API:', bool(probe_api(addr)))

#     data = ['Audi', 132979, 112, 'diesel', 'brown', 'estate',
#             True, True, False, False, True, True, True, 117]
#     data2 = ['model_key', 'mileage', 'engine_power', 'fuel', 'paint_color',
#            'car_type', 'private_parking_available', 'has_gps',
#            'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
#            'has_speed_regulator', 'winter_tires']
    
#     d = {'model_key': 'Audi',
#          'mileage': 132979,
#          'engine_power': 112,
#          'fuel': 'diesel',
#          'paint_color': 'brown',
#          'car_type': 'estate',
#          'private_parking_available': True,
#          'has_gps': True,
#          'has_air_conditioning': False,
#          'automatic_car': False,
#          'has_getaround_connect': True,
#          'has_speed_regulator': True,
#          'winter_tires': True}
#     res = test_predict(d, 'ridge', addr)
    

