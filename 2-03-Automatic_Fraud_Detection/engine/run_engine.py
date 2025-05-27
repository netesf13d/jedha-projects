# -*- coding: utf-8 -*-
"""

"""

import csv
import os

import httpx
import pandas as pd
from sqlalchemy import create_engine, URL, inspect
from sqlalchemy import select
from sqlalchemy.orm import Session

from engine import check_environment_vars
from engine import get_root, get_transaction
from engine import Base, Merchant, Customer, Transaction



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


# !!! check ref tables
## Add merchants and customers table content to database if not present
# merchants_df = pd.read_csv('./merchants_table.csv')
# merchants = [dict(row[1]) for row in merchants_df.iterrows()]
# customers_df = pd.read_csv('./customers_table.csv')
# customers = [dict(row[1]) for row in customers_df.iterrows()]
# with Session(engine) as session:
#     session.add_all(merchants)
#     session.commit()
#     session.add_all(customers)
#     session.commit()


# !!!
## Get last transaction id, initialize to 1 if `transactions` table is empty
transaction_id = 1






# # =============================================================================
# # Get transactions
# # =============================================================================

# try:
#     transaction = get_transaction()
# except httpx.HTTPError as e:
#     print(f"Error fetching transaction: {e}")
# else:
#     transaction = pd.DataFrame(**transaction)





## close database connection gracefully
# engine.dispose()
