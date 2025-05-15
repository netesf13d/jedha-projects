# -*- coding: utf-8 -*-
"""
Classes that represent the tables of the database with SQLAlchemy.
"""

from datetime import datetime, date, time

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
    timestamp: Mapped[datetime] = mapped_column(ForeignKey('times.merchant_id'))
    merchant_id: Mapped[int] = mapped_column(ForeignKey('merchants.merchant_id'))
    customer_id: Mapped[int] = mapped_column(ForeignKey('customers.customer_id'))
    currency_code: Mapped[str] = mapped_column(ForeignKey('currencies.currency_code'))
    ip_geo_id: Mapped[int] = mapped_column(ForeignKey('ip_geography.ip_geo_id'))
    payment_method_id: Mapped[str] = mapped_column(ForeignKey('payment_methods_ref.payment_method_id'))
    payment_status_id: Mapped[str] = mapped_column(ForeignKey('payment_statuses_ref.payment_statuses_id'))
    device_type_id: Mapped[str] = mapped_column(ForeignKey('device_types_ref.device_type_id'))
    amount: Mapped[float]
    
    def __repr__(self):
        str_ = (
            f'Transaction(transaction_id={self.transaction_id!r}, '
            f'timestamp={self.timestamp!r}, '
            f'merchant_id={self.merchant_id!r}, '
            f'customer_id={self.customer_id!r}, '
            f'currency_code={self.currency_code!r}, '
            f'ip_geo_id={self.ip_geo_id!r}, '
            f'payment_method_id={self.payment_method_id!r}, '
            f'payment_status_id={self.payment_status_id!r}, '
            f'device_type_id={self.device_type_id!r}, '
            f'amount={self.amount!r})'
        )
        return str_


class Time(Base):
    """
    Declarative 'times' table structure.
    """
    __tablename__ = 'times'
    
    timestamp: Mapped[datetime] = mapped_column(primary_key=True)
    timezone: Mapped[int]
    date: Mapped[time]
    time: Mapped[time]
    year: Mapped[int]
    month: Mapped[int]
    week_day: Mapped[int]
    quarter: Mapped[int]
    
    def __repr__(self):
        str_ = (
            f'Transaction(timestamp={self.timestamp!r}, '
            f'timezone={self.timezone!r}, '
            f'date={self.date!r}, '
            f'time={self.time!r}, '
            f'year={self.year!r}, '
            f'month={self.month!r}, '
            f'week_day={self.week_day!r}, '
            f'quarter={self.quarter!r})'
        )
        return str_


class Merchant(Base):
    """
    Declarative 'merchants' table structure.
    """
    __tablename__ = 'merchants'
    
    merchant_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    country_code: Mapped[str]
    # ...
    
    def __repr__(self):
        str_ = (
            f'Merchant(merchant_id={self.merchant_id!r}, '
            f'name={self.name!r}, '
            f'country_code={self.country_code!r})'
        )
        return str_
    

class Customer(Base):
    """
    Declarative 'customers' table structure.
    """
    __tablename__ = 'customers'
    
    customer_id: Mapped[int] = mapped_column(primary_key=True)
    country_code: Mapped[str]
    # ...
    
    def __repr__(self):
        str_ = (
            f'Customer(customer_id={self.customer_id!r}, '
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


class IPGeography(Base):
    """
    Declarative 'ip_geography' table structure.
    """
    __tablename__ = 'ip_geography'
    
    ip_geo_id: Mapped[int] = mapped_column(primary_key=True)
    latitude: Mapped[float]
    longitude: Mapped[float]
    country_code: Mapped[int]
    province: Mapped[str]
    # ...
    
    def __repr__(self):
        str_ = (
            f'Merchant(ip_geo_id={self.ip_geo_id!r}, '
            f'latitude={self.latitude!r}, '
            f'longitude={self.longitude!r}, '
            f'country_code={self.country_code!r}, '
            f'province={self.province!r})'
        )
        return str_


# ============================= Reference tables ==============================

class Currency(Base):
    """
    Declarative 'currencies' table structure.
    """
    __tablename__ = 'currencies'
    
    currency_code: Mapped[int] = mapped_column(primary_key=True)
    currency_name: Mapped[str]
    usd_change_rate: Mapped[str]
    # ...
    
    def __repr__(self):
        str_ = (
            f'Merchant(currency_code={self.currency_code!r}, '
            f'currency_name={self.currency_name!r}, '
            f'usd_change_rate={self.usd_change_rate!r})'
        )
        return str_

class Country(Base):
    """
    Declarative 'countries' table structure.
    """
    __tablename__ = 'countries'
    
    country_code: Mapped[int] = mapped_column(primary_key=True)
    country_name: Mapped[str]
    
    def __repr__(self):
        str_ = (
            f'Merchant(country_code={self.country_code!r}, '
            f'country_name={self.country_name!r})'
        )
        return str_


class PaymentMethod(Base):
    """
    Declarative 'payment_methods_ref' table structure.
    """
    __tablename__ = 'payment_methods_ref'
    
    payment_method_id: Mapped[int] = mapped_column(primary_key=True)
    payment_method: Mapped[str]
    
    def __repr__(self):
        str_ = (
            f'Merchant(payment_method_id={self.payment_method_id!r}, '
            f'payment_method={self.payment_method!r})'
        )
        return str_


class PaymentStatus(Base):
    """
    Declarative 'payment_statuses_ref' table structure.
    """
    __tablename__ = 'payment_statuses_ref'
    
    payment_status_id: Mapped[int] = mapped_column(primary_key=True)
    payment_status: Mapped[str]
    
    def __repr__(self):
        str_ = (
            f'Merchant(payment_status_id={self.payment_status_id!r}, '
            f'payment_status={self.payment_status!r})'
        )
        return str_


class DeviceType(Base):
    """
    Declarative 'device_types_ref' table structure.
    """
    __tablename__ = 'device_types_ref'
    
    device_type_id: Mapped[int] = mapped_column(primary_key=True)
    device_type: Mapped[str]
    
    def __repr__(self):
        str_ = (
            f'Merchant(device_type_id={self.device_type_id!r}, '
            f'device_type={self.device_type!r})'
        )
        return str_


class Month(Base):
    """
    Declarative 'months_ref' table structure.
    """
    __tablename__ = 'months_ref'
    
    month: Mapped[int] = mapped_column(primary_key=True)
    month_name: Mapped[str]
    
    def __repr__(self):
        str_ = (
            f'Merchant(month={self.month!r}, '
            f'month_name={self.month_name!r})'
        )
        return str_


class WeekDay(Base):
    """
    Declarative 'week_days_ref' table structure.
    """
    __tablename__ = 'week_days_ref'
    
    week_day: Mapped[int] = mapped_column(primary_key=True)
    week_day_name: Mapped[str]
    
    def __repr__(self):
        str_ = (
            f'Merchant(week_day={self.week_day!r}, '
            f'week_day_name={self.week_day_name!r})'
        )
        return str_