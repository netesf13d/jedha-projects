# -*- coding: utf-8 -*-
"""
!!!
"""

import os

import httpx
import numpy as np
import pandas as pd
from psycopg2.extensions import register_adapter, AsIs
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from engine import check_environment_vars
from engine import get_root, get_transaction, detect_fraud
from engine import Base, Merchant, Customer, Transaction
from engine import (customer_features, merchant_features,
                    fraud_detection_features, transaction_entry)


register_adapter(np.float64, AsIs)
register_adapter(np.int64, AsIs)


FRAUD_MODEL = 'random-forest'


# =============================================================================
# Connect to / create database
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


## setup SQL engine and connect to database
engine = create_engine(os.environ['DATABASE_URI'], echo=False)
Base.metadata.create_all(engine)


## Add merchants and customers table content to database if not present
with Session(engine) as session:
    n_merchants = session.query(Merchant).count()
    n_customers = session.query(Customer).count()

if n_merchants < 693:
    print('Setting up `merchants` table...')
    merchants_df = pd.read_csv('./merchants_table.csv')
    merchants = [Merchant(**dict(row[1])) for row in merchants_df.iterrows()]
    with Session(engine) as session:
        session.add_all(merchants)
        session.commit()
print('Table `merchants` set')

if n_customers < 924:
    print('Setting up `customers` table...')
    customers_df = pd.read_csv('./customers_table.csv')
    customers = [Customer(**dict(row[1])) for row in customers_df.iterrows()]
    with Session(engine) as session:
        session.add_all(customers)
        session.commit()
print('Table `customers` set')

        

## Get last transaction id, initialize to 1 if `transactions` table is empty
statement = select(Transaction.transaction_id)
with Session(engine) as session:
    transact_ids = session.execute(statement).all()
transaction_id = 1 if not transact_ids else max(transact_ids) + 1
print(f'Initialization with transaction id = {transaction_id}')


# =============================================================================
# Get transactions
# =============================================================================

CUSTOMER_COLS = ['cc_num', 'first', 'last', 'gender', 'street', 'city',
                 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob']
MERCHANT_COLS = ['merchant']


try:
    transaction = get_transaction(os.environ['TRANSACTIONS_API_URI'])
except httpx.HTTPError as e:
    print(f"Error fetching transaction: {e}")
else:
    transaction = pd.DataFrame(**transaction).iloc[0]
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

