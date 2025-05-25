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


def test_get_fraud_detection_models(url: str = 'http://localhost:8000')-> list[str]:
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


def test_predict(model_name: str,
                 data: dict[str, str | float | bool],
                 url: str = 'http://localhost:8000')-> dict[str, int]:
    """
    Test the `POST /predict` API endpoint.

    Examples
    --------
    >>> model_name = 'hist-gradient-boosting'
    >>> data = {'month': 12,
    ...         'weekday': 5,
    ...         'day_time': 85664,
    ...         'amt': 419.52,
    ...         'category': 'entertainment',
    ...         'cust_fraudster': True,
    ...         'merch_fraud_victim': True,
    ...         'cos_day_time': 0.998568,
    ...         'sin_day_time': -0.053498}
    >>> test_predict(model_name, data, 'http://localhost:8000')
    {'prediction': 1}
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
    model_names = test_get_fraud_detection_models(addr)
    print('GET fraud_detection_models:', bool(model_names), model_names)

    # /predict endpoint
    data = {'month': 12,
            'weekday': 5,
            'day_time': 85664,
            'amt': 419.52,
            'category': 'entertainment',
            'cust_fraudster': True,
            'merch_fraud_victim': True,
            'cos_day_time': 0.998568,
            'sin_day_time': -0.053498}
    pred = test_predict(model_names[0], data, addr)
    print('POST predict:', bool(pred), pred)


