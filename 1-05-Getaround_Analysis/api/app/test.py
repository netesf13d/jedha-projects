# -*- coding: utf-8 -*-
"""

"""

import sys

import requests


# import mlflow

# =============================================================================
# 
# =============================================================================

def probe_api(url: str = 'http://localhost:8000')-> int:
    """
    Probe the API with the test endpoint.
    """
    r = requests.get(f'{url}/test', timeout=3)
    r.raise_for_status()
    if (res:=r.json()) != 1:
        raise ValueError(f'invalid value: {res}')
    return 1
    

def test_get_pricing_models(url: str = 'http://localhost:8000'
                            )-> dict[str, str]:
    """
    
    """
    r = requests.get(f'{url}/pricing_models', timeout=3)
    r.raise_for_status()
    if not isinstance(res:=r.json(), dict):
        raise ValueError(f'invalid value: {res}')
    return res


def test_get_categories(url: str = 'http://localhost:8000')-> int:
    """
    
    """
    r = requests.get(f'{url}/categories')
    r.raise_for_status()
    if not isinstance(res:=r.json(), dict):
        raise ValueError(f'invalid value: {res}')
    return res


def test_predict(data: list, model: str, url: str = 'http://localhost:8000'):
    r = requests.post(f'{url}/predict', data={'model': 'test'})
    r.raise_for_status()
    return r.json()
    if (res:=r.json()) != 1:
        raise ValueError(f'invalid value: {res}')
    return 1


# d = {'model_key': 'Audi',
#      'mileage': 132979,
#      'engine_power': 112,
#      'fuel': 'diesel',
#      'paint_color': 'brown',
#      'car_type': 'estate',
#      'private_parking_available': True,
#      'has_gps': True,
#      'has_air_conditioning': False,
#      'automatic_car': False,
#      'has_getaround_connect': True,
#      'has_speed_regulator': True,
#      'winter_tires': True}
# r = requests.post("http://localhost:8000/predict", json={'data': d})

tracking_uri = 'https://netesf13d-mlflow-server-1-05-getaround.hf.space/'

# data = {'max_results': 10}
# r = requests.post(uri + 'api/2.0/mlflow/experiments/search', json=data)

# data = {'experiments_ids': [0, 1]}
# r = requests.post(uri + 'api/2.0/mlflow/runs/search', json=data)

from mlflow import MlflowClient

client = MlflowClient(tracking_uri)


models = client.search_registered_models()


# # %%
# experiments = [expt for expt in client.search_experiments()
#                if expt.name != 'Default']

# experiment_ids = [expt.experiment_id for expt in experiments]

# # %%

# runs = [client.search_runs([expt_id])[0] # ], order_by='metrics.r2 DESC')[0]
#         for expt_id in experiment_ids]

# run_ids = [run.info.run_id for run in runs]




# mlflow.set_tracking_uri()


# all_runs = mlflow.search_runs(search_all_experiments=True)




# logged_model = 'runs:/c09d09ef14e546b08f2f339d2c966da6/salary_estimator' 
# model = mlflow.pyfunc.load_model(logged_model)


# if __name__ == '__main__':
#     addr = 'http://localhost:8000'
    
#     print('Probe API:', bool(probe_api(addr)))
#     print('GET models:', bool(res:=test_get_pricing_models(addr)), res)
#     print('GET categories:', bool(res:=test_get_categories(addr)), res)
    

    # data = ['Audi', 132979, 112, 'diesel', 'brown', 'estate',
    #         True, True, False, False, True, True, True, 117]
    # data2 = ['model_key', 'mileage', 'engine_power', 'fuel', 'paint_color',
    #        'car_type', 'private_parking_available', 'has_gps',
    #        'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
    #        'has_speed_regulator', 'winter_tires']
    
    # d = {'model_key': 'Audi',
    #      'mileage': 132979,
    #      'engine_power': 112,
    #      'fuel': 'diesel',
    #      'paint_color': 'brown',
    #      'car_type': 'estate',
    #      'private_parking_available': True,
    #      'has_gps': True,
    #      'has_air_conditioning': False,
    #      'automatic_car': False,
    #      'has_getaround_connect': True,
    #      'has_speed_regulator': True,
    #      'winter_tires': True}
    # res = test_predict(d, 'ridge', addr)
    

