# -*- coding: utf-8 -*-
"""
!!!
Airflow DAGs
Check environment variables and API servers
"""

from airflow.sdk import DAG, Variable, task



@task(task_id='check_variables')
def chk_vars():
    """
    Check that the following variables are set in order to interact with
    the database and APIs:
    - `DATABASE_TOKEN_URI`: the URI of the postgreSQL database
    - `TRANSACTIONS_API_URI`: the url of the API giving new transactions
    - `FRAUD_DETECTION_API_URI`: the url of the fraud detection API
    """
    try:
        Variable.get('AIRFLOW_CONN_TRANSACTION_DB')
    except KeyError:
        raise KeyError('transaction and reference storage database connection '
                       '`AIRFLOW_CONN_TRANSACTION_DB` is not set')
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


@task(task_id='probe_transaction_api')
def probe_transaction_api():
    """
    Check if the transaction API is online.
    """
    from httpx import HTTPError
    from engine_core import probe_transaction_api
    try:
        probe_transaction_api(Variable.get('TRANSACTIONS_API_URI'))
    except HTTPError as e:
        raise HTTPError(f'Transaction API unvailable: {e}')


@task(task_id='probe_fraud_api')
def probe_fraud_api()-> bool:
    """
    Check if the fraud detection API is online.
    """
    from httpx import HTTPError
    from engine_core import probe_fraud_detection_api
    try:
        probe_fraud_detection_api(Variable.get('FRAUD_DETECTION_API_URI'))
    except HTTPError:
        Variable.set(key='fraud_api_online', value=False, serialize_json=True)
    else:
        Variable.set(key='fraud_api_online', value=True, serialize_json=True)


with DAG(
    dag_id='check_environment',
    description=('Check that all the necessary resources are present. '
                 'This DAG is to be triggered manually in case of problem.'),
    max_active_runs=1,
    default_args={'owner': 'admin', 'depends_on_past': False},
    schedule='@once',
    catchup=False,
    tags=['config'],
) as dag:
    chk_vars() >> [probe_transaction_api(), probe_fraud_api()]