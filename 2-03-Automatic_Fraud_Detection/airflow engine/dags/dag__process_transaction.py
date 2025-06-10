# -*- coding: utf-8 -*-
"""
Transaction processing DAG, the core ETL workflow. Proceeds in 4 steps:
- Fetch a transaction event
- Get additional info about the transaction for fraud detection.
- Get a fraud risk score
- Record the transaction in the database
"""

import logging
from datetime import datetime, timedelta

from airflow.sdk import DAG, Variable, task


CUSTOMER_COLS = ['cc_num', 'first', 'last', 'gender', 'street', 'city',
                 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob']
MERCHANT_COLS = ['merchant']



@task(task_id='get_transaction')
def fetch_transaction()-> dict:
    """
    Get a transaction event from the API.
    """
    from httpx import HTTPError
    import pandas as pd
    from engine_core import get_transaction
    try:
        transaction = get_transaction(Variable.get('TRANSACTIONS_API_URI'))
    except HTTPError as e:
        print(f"Error fetching transaction: {e}")

    transact = dict(pd.DataFrame(**transaction).iloc[0])
    for k, v in transact.items():
        transact[k] = getattr(v, 'tolist', lambda: v)()
    return transact


@task(task_id='get_transaction_info')
def get_transaction_info(transaction: dict)-> int:
    """
    Get additional transaction information from the database:
    - `transaction_id`, the id of the transaction
    - merchant id and status (fraud victim or not)
    - customer id and status (fraudster or not)
    """
    from sqlalchemy import create_engine
    from airflow.hooks.base import BaseHook
    from engine_core import (customer_features, merchant_features,
                             transaction_id)

    ## Create engine
    uri = BaseHook.get_connection('transaction_db').get_uri()
    uri = uri.replace('postgres://', 'postgresql://')
    engine = create_engine(uri, echo=False)

    info = {'transaction_id': transaction_id(engine)}
    info.update(merchant_features(transaction, engine))
    info.update(customer_features(transaction, engine))

    ## close the engine gracefully
    engine.dispose()

    return info


@task(task_id='get_fraud_risk')
def get_fraud_risk(transaction: dict,
                   transaction_info: dict,
                   fraud_model: str = 'random-forest')-> bool | None:
    """
    Get the fraud risk associated to the transaction.
    """
    from engine_core import fraud_detection_features, detect_fraud
    try:
        if Variable.get('fraud_api_online', deserialize_json=True):
            pred_features = fraud_detection_features(transaction, transaction_info)
            return detect_fraud(Variable.get('FRAUD_DETECTION_API_URI'),
                                fraud_model, pred_features)
    except:
        Variable.set(key='fraud_api_online', value=False, serialize_json=True)


@task(task_id='record_transaction')
def record_transaction(transaction: dict,
                       transaction_info: dict,
                       fraud_risk: bool | None):
    """
    Store the transaction and the fraud risk in the transaction database.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from airflow.hooks.base import BaseHook
    from engine_core import Transaction
    from engine_core import transaction_entry

    ## Create engine
    uri = BaseHook.get_connection('transaction_db').get_uri()
    uri = uri.replace('postgres://', 'postgresql://')
    engine = create_engine(uri, echo=False)

    ## build entry and store
    entry = transaction_entry(transaction, transaction_info, fraud_risk)
    with Session(engine) as session:
        session.add(Transaction(**entry))
        session.commit()
    logging.info('Record transaction with id '
                 f'{transaction_info["transaction_id"]}')

    ## close the engine gracefully
    engine.dispose()


with DAG(
    dag_id='process_transaction',
    description='Process a transaction: fetch, predict fraud risk, store.',
    max_active_runs=2,
    default_args={'owner': 'admin', 'depends_on_past': False},
    schedule=timedelta(seconds=30),
    start_date=datetime(1970, 1, 1),
    catchup=False,
    tags=['pipeline'],
) as dag:
    _transaction = fetch_transaction()
    _transaction_info = get_transaction_info(_transaction)
    _fraud_risk = get_fraud_risk(_transaction, _transaction_info)
    record_transaction(_transaction, _transaction_info, _fraud_risk)