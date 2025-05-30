# -*- coding: utf-8 -*-
"""
Classes that represent the tables of the database with SQLAlchemy.
"""

from datetime import datetime

from sqlalchemy import Table, ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# =============================================================================
# 
# =============================================================================

class Merchant(Base):
    """
    Declarative 'merchants' table structure.
    """
    __tablename__ = "merchants"
    
    merchant_id: Mapped[int] = mapped_column(primary_key=True)
    merchant: Mapped[str]
    merch_fraud_victim: Mapped[bool]
    
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
    
    customer_id: Mapped[int] = mapped_column(primary_key=True)
    cc_num: Mapped[str]
    first: Mapped[str]
    last: Mapped[str]
    gender: Mapped[str]
    street: Mapped[str]
    city: Mapped[str]
    state: Mapped[str]
    zip: Mapped[int]
    lat: Mapped[float]
    long: Mapped[float]
    city_pop: Mapped[int]
    job: Mapped[str]
    dob: Mapped[str]
    cust_fraudster: Mapped[bool]
    
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
    
    transaction_id: Mapped[int] = mapped_column(primary_key=True)
    merchant_id: Mapped[int] = mapped_column(ForeignKey("merchants.merchant_id"))
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"))
    timestamp: Mapped[datetime]
    month: Mapped[int]
    weekday: Mapped[int]
    day_time: Mapped[float]
    amt: Mapped[float]
    category: Mapped[str]
    cust_fraudster: Mapped[bool]
    merch_fraud_victim: Mapped[bool]
    fraud_risk: Mapped[bool | None]
    
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