# -*- coding: utf-8 -*-
"""
Main dashboard script.
Displays a table showing the last transactions events.
"""

import logging
import os
import time

import streamlit as st
from sqlalchemy import create_engine, inspect
from sqlalchemy import MetaData, Table

from core import make_statement, fetch_data


READ_REPLICA_CONN = os.environ['READ_REPLICA_CONN']
logger = logging.getLogger(__name__)


# =============================================================================
# App initialization
# =============================================================================

########## Setup delay analysis session state ##########
if 'data_available' not in st.session_state:
    engine = create_engine(READ_REPLICA_CONN, echo=False)
    insp = inspect(engine)
    if 'transactions' not in insp.get_table_names():
        logger.error('Table `transactions` does not exist in database')
        st.session_state['data_available'] = False
    else:
        logger.info('Successful connection to database')
        st.session_state['data_available'] = True
        meta = MetaData()
        transactions = Table('transactions', meta, autoload_with=engine)
        st.session_state['transactions'] = transactions
    engine.dispose()

data_available = st.session_state['data_available']
if data_available:
    transactions = st.session_state['transactions']


# =============================================================================
# Application structure
# =============================================================================

########## App configuration ##########
st.set_page_config(
    page_title='Automatic Fraud Detection Dashboard',
    page_icon='\U0001F5D0',
    layout='wide'
)


########## App Title and Description ##########
st.title('Automatic Fraud Detection dashboard \U0001F50D')
st.markdown(
"""
\U0001f44b Hello there! Welcome to this fraud detection dashboard.
The dashboard is connected to the database that sttores the transactions and
displays the last entries from the table.
""")
st.caption('Data provided by [Kaggle]'
           '(https://www.kaggle.com/datasets/kartik2112/fraud-detection/)')


########## Data containers ##########
st.divider()

with st.sidebar:
    limit = st.number_input('Number of transactions displayed',
                            min_value=1, max_value=1000, value=10)

data_container = st.empty()


# =============================================================================
# Dynamically changing content
# =============================================================================

if data_available:
    while True:
        stmt = make_statement(transactions, limit=limit)
        df = fetch_data(READ_REPLICA_CONN, stmt)
        with data_container:
            st.dataframe(df, hide_index=True)
        time.sleep(60)
else:
    st.markdown('### \U000026D4 Database unreachable \U000026D4')