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
    else:
        return dict(pd.DataFrame(**transaction).iloc[0])


@task(task_id='get_transaction_info', multiple_outputs=True)
def get_transaction_info(transaction: dict)-> int:
    """
    !!! doc
    Get last transaction id, initialize to 1 if `transactions` table is empty
    
    """
    from sqlalchemy import create_engine
    from engine_core import Base
    from engine_core import (customer_features, merchant_features,
                             transaction_id)
    
    ## Create engine
    engine = create_engine(Variable.get('AIRFLOW_CONN_TRANSACTION_DB'),
                           echo=False)
    Base.metadata.create_all(engine)
    
    info = {'transaction_id': transaction_id(engine),
            'merch_features': merchant_features(transaction, engine),
            'cust_features': customer_features(transaction, engine)}
    
    engine.dispose()
    
    return info


@task(task_id='get_fraud_risk')
def get_fraud_risk(transaction: dict,
                   merch_features: dict,
                   cust_features: dict,
                   fraud_model: str = 'random-forest')-> bool | None:
    """
    Get the fraud risk associated to the transaction.
    """
    from engine_core import fraud_detection_features, detect_fraud
    
    if Variable.get('fraud_api_online', deserialize_json=True):
        pred_features = fraud_detection_features(
            transaction, merch_features, cust_features)
        return detect_fraud(Variable.get('FRAUD_DETECTION_API_URI'),
                            fraud_model, pred_features)


@task(task_id='record_transaction')
def record_transaction(transaction: dict, transaction_id: int,
                       merch_features: dict, cust_features: dict,
                       fraud_risk: bool | None):
    """
    !!! doc
    Store the transaction and the fraud risk in the transaction database.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from engine_core import Base
    from engine_core import transaction_entry
    
    from numpy import float64, int64
    from psycopg2.extensions import register_adapter, AsIs
    register_adapter(float64, AsIs)
    register_adapter(int64, AsIs)
    
    ## Create engine
    engine = create_engine(Variable.get('AIRFLOW_CONN_TRANSACTION_DB'),
                           echo=False)
    Base.metadata.create_all(engine)
    
    ## build entry and store
    entry = transaction_entry(transaction, transaction_id,
                              merch_features, cust_features, fraud_risk)
    with Session(engine) as session:
        session.add(entry)
        session.commit()
    logging.info(f'Record transaction with id {transaction_id}')
    
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
    _fraud_risk = get_fraud_risk(_transaction_info['merchant_features'],
                                 _transaction_info['customer_features'])
    record_transaction(_transaction, _transaction_info['transaction_id'],
                       _transaction_info['merchant_features'],
                       _transaction_info['customer_features'],
                       _fraud_risk)

