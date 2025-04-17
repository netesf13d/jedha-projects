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

class Transaction(Base):
    """
    Declarative 'transactions' table structure.
    """
    __tablename__ = 'transactions'
    
    transaction_id: Mapped[int] = mapped_column(primary_key=True)
    merchant_id: Mapped[int] = mapped_column(ForeignKey('merchants.merchant_id'))
    customer_id: Mapped[int] = mapped_column(ForeignKey('customers.customer_id'))
    timestamp: Mapped[datetime]
    amount: Mapped[str]
    currency_code: Mapped[str]
    payment_method: Mapped[str]
    payment_status: Mapped[str]
    device_type: Mapped[str]
    ip_latitude: Mapped[float]
    iplongitude: Mapped[float]
    
    def __repr__(self):
        str_ = (
            f'Transaction(transaction_id={self.transaction_id!r}, '
            f'merchant_id={self.merchant_id!r}, '
            f'customer_id={self.customer_id!r}, '
            f'timestamp={self.timestamp!r}, '
            f'amount={self.amount!r}, '
            f'currency_code={self.currency_code!r}, '
            f'payment_method={self.payment_method!r}, '
            f'payment_status={self.payment_status!r}, '
            f'device_type={self.device_type!r}, '
            f'ip_latitude={self.latitude!r}, '
            f'ip_longitude={self.longitude!r})'
        )
        return str_


class Merchant(Base):
    """
    Declarative 'merchants' table structure.
    """
    __tablename__ = 'merchants'
    
    merchant_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    iban: Mapped[str]
    country_code: Mapped[str]
    # ...
    
    def __repr__(self):
        str_ = (
            f'Merchant(merchant_id={self.merchant_id!r}, '
            f'name={self.name!r}, '
            f'iban={self.iban!r}, '
            f'country_code={self.country_code!r})'
        )
        return str_
    

class Customer(Base):
    """
    Declarative 'customers' table structure.
    """
    __tablename__ = 'customers'
    
    customer_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    iban: Mapped[str]
    country_code: Mapped[str]
    # ...
    
    def __repr__(self):
        str_ = (
            f'Customer(customer_id={self.customer_id!r}, '
            f'name={self.name!r}, '
            f'iban={self.iban!r}, '
            f'country_code={self.country_code!r})'
        )
        return str_


class FraudIndicator(Base):
    """
    Declarative 'fraud_indicators' table structure.
    """
    __tablename__ = 'fraud_indicators'
    
    transaction_id: Mapped[int] = mapped_column(primary_key=True)
    fraud_probability: Mapped[float | None]
    # ...
    
    def __repr__(self):
        str_ = (
            f'FraudIndicator(transaction_id={self.transaction_id!r}, '
            f'fraud_probability={self.fraud_probability!r}, '
        )
        return str_



def reflect_oltp_db(metadata_obj, engine)-> tuple[Table, Table, Table, Table]:
    """
    Create tables by loading information from database.
    """
    transactions = Table('transactions', metadata_obj, autoload_with=engine)
    merchants = Table('merchants', metadata_obj, autoload_with=engine)
    customers = Table('customers', metadata_obj, autoload_with=engine)
    fraud_indicators = Table('fraud_indicators', metadata_obj,
                             autoload_with=engine)
    return (transactions, merchants, customers, fraud_indicators)