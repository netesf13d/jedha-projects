# -*- coding: utf-8 -*-
"""
ETL processing utilities.
"""

import logging
from datetime import datetime

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .db_mgmt_14 import Customer, Merchant, Transaction


CUSTOMER_COLS = ['cc_num', 'first', 'last', 'gender', 'street', 'city',
                 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob']
MERCHANT_COLS = ['merchant']

# =============================================================================
#
# =============================================================================

def transaction_id(engine)-> int:
    """
    Get the transaction id of the next transaction.
    """
    statement = select(func.max(Transaction.transaction_id))
    with Session(engine) as session:
        transact_ids = session.execute(statement).all()[0][0]
    return 1 if transact_ids is None else transact_ids + 1


def merchant_features(transaction: dict, engine)-> dict:
    """
    !!!

    Parameters
    ----------
    transaction : pd.Series
        DESCRIPTION.
    engine : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    merch_filt = {col: transaction[col] for col in MERCHANT_COLS}
    merch_filt['merchant'] = merch_filt['merchant'].strip('fraud_')

    statement = (
        select(Merchant.merchant_id, Merchant.merch_fraud_victim)
        .filter_by(**merch_filt)
    )
    with Session(engine) as session:
        merch_features = session.execute(statement).all()
    
    try:
        merch_features = merch_features[0]
    except IndexError:
        logging.error(f'no merchant corresponding to {merch_filt}')
        raise
    
    return {'merchant_id': merch_features[0],
            'merch_fraud_victim': merch_features[1]}


def customer_features(transaction: dict, engine)-> dict:
    """
    !!!

    Parameters
    ----------
    transaction : pd.Series
        DESCRIPTION.
    engine : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    cust_filt = {col: transaction[col] for col in CUSTOMER_COLS}
    cust_filt['cc_num'] = str(cust_filt['cc_num'])

    statement = (
        select(Customer.customer_id, Customer.cust_fraudster)
        .filter_by(**cust_filt)
    )
    with Session(engine, future=True) as session:
        cust_features = session.execute(statement).all()
    
    try:
        cust_features = cust_features[0]
    except IndexError:
        logging.error(f'no merchant corresponding to {cust_filt}')
        raise
    
    return {'customer_id': cust_features[0],
            'cust_fraudster': cust_features[1]}


def fraud_detection_features(transaction: dict,
                             transaction_info: dict)-> dict:
    """
    !!!

    Parameters
    ----------
    transaction : pd.Series
        DESCRIPTION.
    merchant_features : dict
        DESCRIPTION.
    customer_features : dict
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    ts = transaction['current_time']//1000
    day_time = ts % 86400
    ts = datetime.fromtimestamp(ts)
    features = {
        'month': ts.month,
        'weekday': ts.weekday(),
        'day_time': day_time,
        'amt': transaction['amt'],
        'category': transaction['category'],
        'cust_fraudster': transaction_info['cust_fraudster'],
        'merch_fraud_victim': transaction_info['merch_fraud_victim'],
        'cos_day_time': float(np.cos(2*np.pi*day_time/86400)),
        'sin_day_time': float(np.sin(2*np.pi*day_time/86400)),
        }

    return features


def transaction_entry(transaction: dict,
                      transaction_info: dict,
                      fraud_risk: bool)-> dict:
    """
    !!!

    Parameters
    ----------
    transaction : pd.Series
        DESCRIPTION.
    merchant_features : dict
        DESCRIPTION.
    customer_features : dict
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    ts = transaction['current_time']//1000
    day_time = ts % 86400
    ts = datetime.fromtimestamp(ts)

    entry = {'timestamp': ts.isoformat(),
             'month': ts.month,
             'weekday': ts.weekday(),
             'day_time': day_time,
             'amt': transaction['amt'],
             'category': transaction['category'],
             'fraud_risk': fraud_risk}
    entry.update(transaction_info)

    return entry


