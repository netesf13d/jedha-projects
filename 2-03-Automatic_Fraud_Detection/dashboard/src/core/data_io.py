# -*- coding: utf-8 -*-
"""
Utilities to fetch data from the database.
"""

import pandas as pd
from sqlalchemy import create_engine, select, Table


def make_statement(transactions: Table, limit: int = 10):
    """
    Construct the select statement.
    """
    statement = (
        select(transactions)
        .order_by(transactions.c.timestamp.desc())
        .limit(limit)
    )
    return statement


def fetch_data(conn_uri: str, statement)-> pd.DataFrame:
    """
    Fetch the data from the database and store it in a dataframe.
    """
    engine = create_engine(conn_uri, echo=False)
    with engine.begin() as conn:
        df = pd.read_sql_query(statement, conn)
    engine.dispose()
    return df



