# -*- coding: utf-8 -*-
"""
!!!
Airflow DAGs

Process transactions DAG
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
    !!! doc
    Get last transaction id, initialize to 1 if `transactions` table is empty
    
    """
    from sqlalchemy import create_engine
    from engine_core import Base, Merchant, Customer
    from engine_core import (customer_features, merchant_features,
                             transaction_id)
    
    ## Create engine
    engine = create_engine(Variable.get('AIRFLOW_CONN_TRANSACTION_DB'),
                           echo=False)
    Base.metadata.create_all(engine)

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
    
    if Variable.get('fraud_api_online', deserialize_json=True):
        pred_features = fraud_detection_features(transaction, transaction_info)
        return detect_fraud(Variable.get('FRAUD_DETECTION_API_URI'),
                            fraud_model, pred_features)


@task(task_id='record_transaction')
def record_transaction(transaction: dict,
                       transaction_info: dict,
                       fraud_risk: bool | None):
    """
    !!! doc
    Store the transaction and the fraud risk in the transaction database.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from engine_core import Base, Transaction
    from engine_core import transaction_entry
    
    ## Create engine
    engine = create_engine(Variable.get('AIRFLOW_CONN_TRANSACTION_DB'),
                           echo=False)
    Base.metadata.create_all(engine)
    
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
    max_active_runs=1,
    default_args={'owner': 'admin', 'depends_on_past': False},
    schedule=timedelta(seconds=20),
    start_date=datetime(1970, 1, 1),
    catchup=False,
    tags=['pipeline'],
) as dag:
    _transaction = fetch_transaction()
    _transaction_info = get_transaction_info(_transaction)
    _fraud_risk = get_fraud_risk(_transaction, _transaction_info)
    record_transaction(_transaction, _transaction_info, _fraud_risk)

