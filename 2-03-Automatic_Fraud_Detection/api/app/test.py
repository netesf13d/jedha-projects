# -*- coding: utf-8 -*-
"""
File containing tests for the API endpoints.
"""

import requests



def probe_api(url: str = 'http://localhost:8000')-> int:
    """
    Probe the API with the `GET /test` endpoint.
    Returns 1 or raises ValueError.

    Examples
    --------
    >>> probe_api('http://localhost:8000')
    1
    """
    r = requests.get(f'{url}/test', timeout=3)
    r.raise_for_status()
    if (res:=r.json()) != 1:
        raise ValueError(f'invalid value: {res}')
    return res


def test_get_pricing_models(url: str = 'http://localhost:8000')-> list[str]:
    """
    Test the `GET /get_pricing_models` API endpoint.

    Examples
    --------
    >>> test_get_pricing_models('http://localhost:8000')
    ['ridge-regression', 'gradient-boosting']
    """
    r = requests.get(f'{url}/pricing_models', timeout=3)
    r.raise_for_status()
    if not isinstance(res:=r.json(), list):
        raise ValueError(f'invalid value: {r.content}')
    return res


def test_get_categories(url: str = 'http://localhost:8000')-> int:
    """
    Test the `GET /get_categories` API endpoint.

    Examples
    --------
    >>> test_get_categories('http://localhost:8000')
    {'model_key': ['Alfa Romeo', 'Audi', 'BMW', ...],
     'fuel': ['diesel', 'electro', 'hybrid_petrol', 'petrol'],
     'paint_color': ['beige', 'black', ...],
     'car_type': ['convertible', 'coupe', ...]}
    """
    r = requests.get(f'{url}/categories')
    r.raise_for_status()
    if not isinstance(res:=r.json(), dict):
        raise ValueError(f'invalid value: {r.content}')
    return res


def test_predict(model_name: str,
                 data: dict[str, str | float | bool],
                 url: str = 'http://localhost:8000')-> dict[str, float]:
    """
    Test the `POST /predict` API endpoint.

    Examples
    --------
    >>> model_name = 'ridge_regression'
    >>> data = {'model_key': 'Audi',
    ...         'mileage': 132979,
    ...         'engine_power': 112,
    ...         'fuel': 'diesel',
    ...         'paint_color': 'brown',
    ...         'car_type': 'estate',
    ...         'private_parking_available': True,
    ...         'has_gps': True,
    ...         'has_air_conditioning': False,
    ...         'automatic_car': False,
    ...         'has_getaround_connect': True,
    ...         'has_speed_regulator': True,
    ...         'winter_tires': True}
    >>> test_predict(model_name, data, 'http://localhost:8000')
    {'prediction': 120.81970398796342}
    """

    r = requests.post(f'{url}/predict/{model_name}', json=data)
    r.raise_for_status()
    if not isinstance(res:=r.json(), dict):
        raise ValueError(f'invalid value: {r.content}')
    return res


if __name__ == '__main__':
    addr = 'http://localhost:8000'

    # /test endpoint
    print('Probe API:', bool(probe_api(addr)))

    # /pricing_models endpoint
    model_names = test_get_pricing_models(addr)
    print('GET pricing_models:', bool(model_names), model_names)

    # /categories endpoint
    print('GET categories:', bool(res:=test_get_categories(addr)), res)

    # /predict endpoint
    data = {"model_key": "Audi",
            "mileage": 132979,
            "engine_power": 112,
            "fuel": "diesel",
            "paint_color": "brown",
            "car_type": "estate",
            "private_parking_available": True,
            "has_gps": True,
            "has_air_conditioning": False,
            "automatic_car": False,
            "has_getaround_connect": True,
            "has_speed_regulator": True,
            "winter_tires": True}
    pred = test_predict(model_names[0], data, addr)
    print('POST predict:', bool(pred), pred)


