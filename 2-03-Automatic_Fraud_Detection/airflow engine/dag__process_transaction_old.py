# -*- coding: utf-8 -*-
"""
!!!
Airflow DAGs
"""

import logging
from datetime import datetime, timedelta

from airflow.sdk import DAG, Variable, task, task_group
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator



CUSTOMER_COLS = ['cc_num', 'first', 'last', 'gender', 'street', 'city',
                 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob']
MERCHANT_COLS = ['merchant']


# =============================================================================
# Process transactions
# =============================================================================


@task(task_id='get_info', multiple_outputs=True)
def get_info(engine, transaction: dict)-> int:
    """
    !!! REDO
    Get last transaction id, initialize to 1 if `transactions` table is empty
    
    SELECT
      MAX(transaction_id)
    FROM
      transactions
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





@task(task_id='get_transaction_id')
def get_transaction_id(engine)-> int:
    """
    !!! REDO
    Get last transaction id, initialize to 1 if `transactions` table is empty
    
    SELECT
      MAX(transaction_id)
    FROM
      transactions
    """
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session
    from engine_core import Transaction
    
    statement = select(func.max(Transaction.transaction_id))
    with Session(engine) as session:
        transact_ids = session.execute(statement).all()[0][0]
    transaction_id = 1 if transact_ids is None else max(transact_ids) + 1
    return transaction_id


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


# @task(task_id='get_merchant_features', multiple_outputs=True)
# def get_merchant_features(transaction: dict)-> dict[str, dict]:
#     """
#     !!! REDO
#     Fetch merchant features for fraud detection.
#     """
#     merch_filt = {col: transaction[col] for col in MERCHANT_COLS}
#     merch_filt['merchant'] = merch_filt['merchant'].strip('fraud_')

    # stmt = ("SELECT merchant_id, merch_fraud_victim"
    #         "  FROM merchants WHERE"
    #         + " AND ". join(f'{k} = "{v}"' for k, v in merch_filt))
    
#     statement = (
#         select(Merchant.merchant_id, Merchant.merch_fraud_victim)
#         .filter_by(**merch_filt)
#     )
#     with Session(engine) as session:
#         merch_features = session.execute(statement).all()[0]
    
#     return {'merchant_id': merch_features[0],
#             'merch_fraud_victim': merch_features[1]}
    
#     # from engine_core import customer_features, merchant_features
    
#     # return {'merch_features': merchant_features(transaction, engine),
#     #         'cust_features': customer_features(transaction, engine)}


# @task(task_id='get_customer_features', multiple_outputs=True)
# def get_customer_features(transaction: dict)-> dict[str, dict]:
#     """
#     !!! REDO
#     Fetch customer features for fraud detection.
    
#     SELECT
#       MAX(transaction_id)
#     FROM
#       transactions
#     """
#     cust_filt = {col: transaction[col] for col in CUSTOMER_COLS}
#     cust_filt['cc_num'] = str(cust_filt['cc_num'])

#     stmt = ("SELECT customer_id, merch_fraud_victim"
#             "  FROM customers WHERE"
#             + " AND ". join(f'{k} = "{v}"' for k, v in cust_filt))

#     statement = (
#         select(Merchant.merchant_id, Merchant.merch_fraud_victim)
#         .filter_by(**merch_filt)
#     )
#     with Session(engine, future=True) as session:
#         cust_features = session.execute(statement).all()[0]
    
#     return {'customer_id': cust_features[0],
#             'cust_fraudster': cust_features[1]}
    
#     # from engine_core import customer_features, merchant_features
    
#     # return {'merch_features': merchant_features(transaction, engine),
#     #         'cust_features': customer_features(transaction, engine)}


# @task(task_id='get_user_entries')
# def get_user_entries(transaction: dict,
#                      user_columns: list[str])-> dict[str, dict]:
#     """
#     !!! REDO
#     Fetch merchant features for fraud detection.
#     """
#     filt = {col: transaction[col] for col in user_columns}
#     try:
#         filt['merchant'] = filt['merchant'].strip('fraud_')
#     except KeyError:
#         pass
#     return filt

@task(task_id='get_merchant_query')
def get_merchant_query(transaction: dict)-> str:
    """
    !!! REDO
    Fetch merchant features for fraud detection.
    """
    merch_filt = {col: transaction[col] for col in MERCHANT_COLS}
    merch_filt['merchant'] = merch_filt['merchant'].strip('fraud_')
    query = ("SELECT merchant_id, merch_fraud_victim"
             "  FROM merchants WHERE"
             + " AND ". join(f'{k} = "{v}"' for k, v in merch_filt))
    return query


@task_group(group_id='get_merchant_features')
def get_merchant_features(transaction: dict)-> dict[str, dict]:
    """
    !!! REDO
    Fetch customer features for fraud detection.
    
    SELECT
      MAX(transaction_id)
    FROM
      transactions
    """
    merch_query = get_merchant_query(transaction)
    
    query_res = SQLExecuteQueryOperator(
        task_id='query_customer_features',
        conn_id='transaction_db',
        sql='%(query)',
        parameters={'query': merch_query},
    )
    
    merch_query >> query_res



@task(task_id='get_customer_query')
def get_customer_query(transaction: dict)-> str:
    """
    !!! REDO
    Fetch merchant features for fraud detection.
    """
    cust_filt = {col: transaction[col] for col in CUSTOMER_COLS}
    query = ('SELECT customer_id, cust_fraudster'
             '  FROM customers WHERE'
             + ' AND '. join(f'{k} = "{v}"' for k, v in cust_filt))
    return query


@task_group(group_id='get_customer_features')
def get_customer_features(transaction: dict)-> dict[str, dict]:
    """
    !!! REDO
    Fetch customer features for fraud detection.
    
    SELECT
      MAX(transaction_id)
    FROM
      transactions
    """
    cust_query = get_customer_query(transaction)
    
    query_res = SQLExecuteQueryOperator(
        task_id='query_customer_features',
        conn_id='transaction_db',
        sql='%(query)',
        parameters={'query': cust_query},
    )
    
    cust_query >> query_res
    
    # statement = (
    #     select(Merchant.merchant_id, Merchant.merch_fraud_victim)
    #     .filter_by(**merch_filt)
    # )
    # with Session(engine, future=True) as session:
    #     cust_features = session.execute(statement).all()[0]
    
    # return {'customer_id': cust_features[0],
    #         'cust_fraudster': cust_features[1]}


# @task_group(group_id='get_customer_features')
# def get_customer_features(transaction: dict)-> dict[str, dict]:
#     """
#     !!! REDO
#     Fetch customer features for fraud detection.
    
#     SELECT
#       MAX(transaction_id)
#     FROM
#       transactions
#     """
#     cust_filt = get_user_entries(transaction, CUSTOMER_COLS)
    
    
#     cust_filt = {col: transaction[col] for col in CUSTOMER_COLS}
#     cust_filt['cc_num'] = str(cust_filt['cc_num'])

#     stmt = ("SELECT customer_id, merch_fraud_victim"
#             "  FROM customers WHERE"
#             + " AND ". join(f'{k} = "{v}"' for k, v in cust_filt))

#     statement = (
#         select(Merchant.merchant_id, Merchant.merch_fraud_victim)
#         .filter_by(**merch_filt)
#     )
#     with Session(engine, future=True) as session:
#         cust_features = session.execute(statement).all()[0]
    
#     return {'customer_id': cust_features[0],
#             'cust_fraudster': cust_features[1]}
    
    # from engine_core import customer_features, merchant_features
    
    # return {'merch_features': merchant_features(transaction, engine),
    #         'cust_features': customer_features(transaction, engine)}


with DAG(
    dag_id='test',
    max_active_runs=1,
    default_args={'owner': 'admin', 'depends_on_past': False},
    schedule=None,
    start_date=datetime(1970, 1, 1),
    catchup=False,
    tags=['pipeline'],
) as dag:
    
    query_res = SQLExecuteQueryOperator(
        task_id='query_customer_features',
        conn_id='transaction_db',
        sql='SELECT MAX(transaction_id) FROM transactions',
    )
    
    query_res
    
    # _transaction_id = get_transaction_id()
    # _transaction = fetch_transaction()
    # _merch_features = get_merchant_features(_transaction)
    # _cust_features = get_customer_features(_transaction)
    # _fraud_risk = get_fraud_risk(_merch_features, _cust_features)
    # record_transaction(_transaction, _transaction_id,
    #                    _merch_features, _cust_features ,_fraud_risk)
















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
def record_transaction(engine,
                       transaction: dict, transaction_id: int,
                       merch_features: dict, cust_features: dict,
                       fraud_risk: bool | None):
    """
    !!! REDO
    Store the transaction and the fraud risk in the transaction database.
    """
    from sqlalchemy.orm import Session
    from engine_core import transaction_entry
    
    import numpy as np
    from psycopg2.extensions import register_adapter, AsIs
    register_adapter(np.float64, AsIs)
    register_adapter(np.int64, AsIs)
    
    transaction_out = transaction_entry(transaction, transaction_id,
                                        merch_features, cust_features,
                                        fraud_risk)
    with Session(engine) as session:
        session.add(transaction_out)
        session.commit()
        transaction_id += 1
    logging.info(f'Record transaction with id {transaction_id}')


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
    
    _transaction_id = get_transaction_id()
    _transaction = fetch_transaction()
    _merch_features = get_merchant_features(_transaction)
    _cust_features = get_customer_features(_transaction)
    _fraud_risk = get_fraud_risk(_merch_features, _cust_features)
    record_transaction(_transaction, _transaction_id,
                       _merch_features, _cust_features ,_fraud_risk)

