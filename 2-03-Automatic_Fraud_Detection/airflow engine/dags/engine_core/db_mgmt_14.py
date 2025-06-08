# -*- coding: utf-8 -*-
"""
Classes that represent the tables of the database with SQLAlchemy.
This version is adapter to SQLAlchemy==1.4, compatible with Airflow==3.0
"""

from datetime import datetime

from sqlalchemy import Table, ForeignKey, Column
from sqlalchemy import Integer, String, Boolean, Float, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Mapped


Base = declarative_base()


# =============================================================================
#
# =============================================================================

class Merchant(Base):
    """
    Declarative 'merchants' table structure.
    """
    __tablename__ = "merchants"

    merchant_id: Mapped[int] = Column(Integer, primary_key=True)
    merchant: Mapped[str] = Column(String)
    merch_fraud_victim: Mapped[bool] = Column(Boolean)

    def __repr__(self):
        str_ = (
            f"Merchant(merchant_id={self.merchant_id!r}, "
            f"merchant={self.merchant!r}, "
            f"merch_fraud_victim={self.merch_fraud_victim!r})"
        )
        return str_


class Customer(Base):
    """
    Declarative 'customers' table structure.
    """
    __tablename__ = "customers"

    customer_id: Mapped[int] = Column(Integer, primary_key=True)
    cc_num: Mapped[str] = Column(String)
    first: Mapped[str] = Column(String)
    last: Mapped[str] = Column(String)
    gender: Mapped[str] = Column(String)
    street: Mapped[str] = Column(String)
    city: Mapped[str] = Column(String)
    state: Mapped[str] = Column(String)
    zip: Mapped[int] = Column(Integer)
    lat: Mapped[float] = Column(Float)
    long: Mapped[float] = Column(Float)
    city_pop: Mapped[int] = Column(Integer)
    job: Mapped[str] = Column(String)
    dob: Mapped[str] = Column(String)
    cust_fraudster: Mapped[bool] = Column(Boolean)

    def __repr__(self):
        str_ = (
            f"Customer(customer_id={self.customer_id!r}, "
            f"cc_num={self.cc_num!r}, "
            f"first={self.first!r}, "
            f"last={self.last!r}, "
            f"gender={self.gender!r}, "
            f"street={self.street!r}, "
            f"city={self.city!r}, "
            f"state={self.state!r}, "
            f"zip={self.zip!r}, "
            f"lat={self.lat!r}, "
            f"lon={self.lon!r}, "
            f"city_pop={self.city_pop!r}, "
            f"job={self.job!r}, "
            f"dob={self.dob!r}, "
            f"cust_fraudster={self.cust_fraudster!r})"
        )
        return str_


class Transaction(Base):
    """
    Declarative 'transactions' table structure.
    """
    __tablename__ = 'transactions'

    transaction_id: Mapped[int] = Column(Integer, primary_key=True)
    merchant_id: Mapped[int] = Column(Integer, ForeignKey("merchants.merchant_id"))
    customer_id: Mapped[int] = Column(Integer, ForeignKey("customers.customer_id"))
    timestamp: Mapped[datetime] = Column(DateTime)
    month: Mapped[int] = Column(Integer)
    weekday: Mapped[int] = Column(Integer)
    day_time: Mapped[float] = Column(Float)
    amt: Mapped[float] = Column(Float)
    category: Mapped[str] = Column(String)
    cust_fraudster: Mapped[bool] = Column(Boolean)
    merch_fraud_victim: Mapped[bool] = Column(Boolean)
    fraud_risk: Mapped[bool | None] = Column(Boolean, nullable=True)

    def __repr__(self):
        str_ = (
            f"Transaction(transaction_id={self.transaction_id!r}, "
            f"merchant_id={self.merchant_id!r}, "
            f"customer_id={self.customer_id!r}, "
            f"timestamp={self.timestamp!r}, "
            f"month={self.month!r}, "
            f"weekday={self.weekday!r}, "
            f"day_time={self.day_time!r}, "
            f"amt={self.amt!r}, "
            f"category={self.category!r}, "
            f"cust_fraudster={self.cust_fraudster!r}, "
            f"merch_fraud_victim={self.merch_fraud_victim!r}, "
            f"fraud_risk={self.fraud_risk!r})"
        )
        return str_


def reflect_db(metadata_obj, engine)-> tuple[Table, Table, Table]:
    """
    Create tables by loading information from database.
    """
    merchants = Table('merchants', metadata_obj, autoload_with=engine)
    customers = Table('customers', metadata_obj, autoload_with=engine)
    transactions = Table('transactions', metadata_obj, autoload_with=engine)
    return (merchants, customers, transactions)