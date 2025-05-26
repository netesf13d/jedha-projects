# -*- coding: utf-8 -*-
"""

"""

import os

import httpx
import pandas as pd
from sqlalchemy import create_engine, URL, inspect
from sqlalchemy import select
from sqlalchemy.orm import Session

from engine import get_root, get_transaction
from engine import Base, Merchant, Customer, Transaction


API_ = 'https://charlestng-real-time-fraud-detection.hf.space/'
API2 = 'https://netesf13d-api-2-03-fraud-detection.hf.space/'


# =============================================================================
# Connect / create database
# =============================================================================

db_url = ''

## setup SQL engine
engine = create_engine(db_url, echo=False)
# inspector = inspect(engine)

## clear the database
# Base.metadata.drop_all(engine)

## create the database
Base.metadata.create_all(engine)



## Transfer to database
Base.metadata.create_all(engine)
with Session(engine) as session:
    session.add_all(locations)
    session.commit()
    session.add_all(weather_indicators)
    session.commit()
    session.add_all(hotels)
    session.commit()




# =============================================================================
# Get transactions
# =============================================================================

try:
    transaction = get_transaction()
except httpx.HTTPError as e:
    print(f"Error fetching transaction: {e}")
else:
    transaction = pd.DataFrame(**transaction)






