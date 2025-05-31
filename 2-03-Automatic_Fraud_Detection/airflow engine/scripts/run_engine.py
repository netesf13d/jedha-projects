# -*- coding: utf-8 -*-
"""
!!!
Run the fraud detection pipeline with a single transaction
"""

import os
import sys

sys.path.append('..')

import httpx
import numpy as np
import pandas as pd
from psycopg2.extensions import register_adapter, AsIs
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session

from core import check_environment_vars
from core import (probe_transaction_api, get_transaction,
                  probe_fraud_detection_api, detect_fraud)
from core import Base, Merchant, Customer, Transaction
from core import (customer_features, merchant_features,
                    fraud_detection_features, transaction_entry)


register_adapter(np.float64, AsIs)
register_adapter(np.int64, AsIs)


FRAUD_MODEL = 'random-forest'


# =============================================================================
# Check environment variables and API servers
# =============================================================================

os.environ['TRANSACTIONS_API_URI'] = 'https://charlestng-real-time-fraud-detection.hf.space/'
os.environ['FRAUD_DETECTION_API_URI'] = 'https://netesf13d-api-2-03-fraud-detection.hf.space/'
# Try to get the database uri from a file
try:
    with open('fraud-detection-ref-db.key', 'rt', encoding='utf-8') as f:
        os.environ['DATABASE_URI'] = f.read()
except FileNotFoundError:
    pass
check_environment_vars()


# =============================================================================
# Connect to / create database
# =============================================================================

## setup SQL engine and connect to database
engine = create_engine(os.environ['DATABASE_URI'], echo=False)
Base.metadata.create_all(engine)


## Add merchants and customers table content to database if not present
with Session(engine) as session:
    n_merchants = session.query(Merchant).count()
    n_customers = session.query(Customer).count()

if n_merchants < 693: # merchants table absent from database
    print('Setting up `merchants` table...')
    merchants_df = pd.read_csv('./merchants_table.csv')
    merchants = [Merchant(**dict(row[1])) for row in merchants_df.iterrows()]
    with Session(engine) as session:
        session.add_all(merchants)
        session.commit()
print('Table `merchants` set')

if n_customers < 924: # customers table absent from database
    print('Setting up `customers` table...')
    customers_df = pd.read_csv('./customers_table.csv')
    customers = [Customer(**dict(row[1])) for row in customers_df.iterrows()]
    with Session(engine) as session:
        session.add_all(customers)
        session.commit()
print('Table `customers` set')

        

## Get last transaction id, initialize to 1 if `transactions` table is empty
statement = select(func.max(Transaction.transaction_id))
with Session(engine) as session:
    transact_ids = session.execute(statement).all()[0][0]
transaction_id = 1 if transact_ids is None else max(transact_ids) + 1
print(f'Initialization with transaction id = {transaction_id}')


# =============================================================================
# Process transactions
# =============================================================================

try:
    transaction = get_transaction(os.environ['TRANSACTIONS_API_URI'])
except httpx.HTTPError as e:
    print(f"Error fetching transaction: {e}")
else:
    transaction = dict(pd.DataFrame(**transaction).iloc[0])
print(transaction)

merch_features = merchant_features(transaction, engine)
cust_features = customer_features(transaction, engine)

pred_features = fraud_detection_features(
    transaction, merch_features, cust_features)

fraud_risk = detect_fraud(os.environ['FRAUD_DETECTION_API_URI'],
                          FRAUD_MODEL, pred_features)

transaction_out = transaction_entry(transaction, transaction_id,
                                    merch_features, cust_features,
                                    fraud_risk)
with Session(engine) as session:
    session.add(transaction_out)
    session.commit()
    transaction_id += 1

## close the engine gracefully
engine.dispose()

