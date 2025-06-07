# -*- coding: utf-8 -*-
"""
!!!
Airflow DAGs
"""

import logging
from datetime import datetime, timedelta

from airflow.sdk import DAG, task, Variable


# =============================================================================
# Shared tasks
# =============================================================================

@task(task_id='connect_to_db')
def connect_to_database():
    """
    !!!
    """
    from sqlalchemy import create_engine
    from engine_core import Base
    
    engine = create_engine(Variable.get('DATABASE_URI'), echo=False)
    Base.metadata.create_all(engine)
    return engine


@task(task_id='close_connection')
def close_engine(engine):
    """
    Close the engine gracefully.
    """
    engine.dispose()


# =============================================================================
# Check environment variables and API servers
# =============================================================================

@task(task_id='chk_env_vars2')
def check_env_vars2():
    """
    Check that the following variables are set in order to interact with
    the database and APIs:
    - `DATABASE_URI`: the URI of the postgreSQL database
    - `TRANSACTIONS_API_URI`: the url of the API giving new transactions
    - `FRAUD_DETECTION_API_URI`: the url of the fraud detection API
    """
    try:
        Variable.get('DATABASE_URI')
    except KeyError:
        raise KeyError('transaction storage database variable '
                       '`DATABASE_URI` is not set')
    try:
        Variable.get('TRANSACTIONS_API_URI')
    except KeyError:
        raise KeyError('transactions API url environment variable '
                       '`TRANSACTIONS_API_URI` is not set')
    try:
        Variable.get('FRAUD_DETECTION_API_URI')
    except KeyError:
        raise KeyError('fraud detection API url variable '
                       '`FRAUD_DETECTION_API_URI` is not set')
    


@task(task_id='probe_transact_api2')
def probe_transact_api2():
    """
    !!!
    """
    from httpx import HTTPError
    from engine_core import probe_transaction_api
    try:
        probe_transaction_api(Variable.get('TRANSACTIONS_API_URI'))
    except HTTPError as e:
        raise HTTPError(f'Transaction API unvailable: {e}')


@task(task_id='probe_fraud_api2')
def probe_fraud_api2()-> bool:
    """
    !!!
    """
    from httpx import HTTPError
    from engine_core import probe_fraud_detection_api
    try:
        probe_fraud_detection_api(Variable.get('FRAUD_DETECTION_API_URI'))
    except HTTPError:
        Variable.set(key='fraud_api_online', value=False, serialize_json=True)
    else:
        Variable.set(key='fraud_api_online', value=True, serialize_json=True)
        
    print(Variable.get(key='fraud_api_online', deserialize_json=True))
    logging.info(Variable.get(key='fraud_api_online', deserialize_json=True))
    


with DAG(
    dag_id='check_environment2',
    description=('Check that all the necessary resources are present. '
                 'This DAG is to be triggered manually in case of problem.'),
    max_active_runs=1,
    default_args={'owner': 'admin', 'depends_on_past': False},
    schedule=None,
    catchup=False,
    tags=['config'],
) as dag:
    
    check_env_vars2() >> [probe_transact_api2(), probe_fraud_api2()]

