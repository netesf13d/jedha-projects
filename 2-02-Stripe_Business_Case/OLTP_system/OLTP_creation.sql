CREATE DATABASE stripe_oltp

CREATE SCHEMA transactions AUTHORIZATION oltpadmin;
GO


CREATE TABLE transactions.transactions (
  transaction_id BIGINT PRIMARY KEY
  merchant_id BIGINT FOREIGN KEY
  customer_id BIGINT FOREIGN KEY
  timestamp DATETIME
  amount FLOAT
  fee FLOAT
  currency_code CHAR(3) -- ISO 4217
  payment_method TEXT
  payment_status TEXT
  device_type TEXT
  ip_latitude FLOAT
  ip_longitude FLOAT
);

CREATE TABLE transactions.merchants (
  merchant_id BIGINT PRIMARY KEY
  name TEXT
  iban TEXT
  country_code CHAR(2) -- ISO 3166-1 alpha-2
  -- ...
);

CREATE TABLE transactions.customers (
  customer_id BIGINT PRIMARY KEY
  name TEXT
  iban TEXT
  country_code CHAR(2) -- ISO 3166-1 alpha-2
  -- ...
);

CREATE TABLE transactions.fraud_indicators (
  transaction_id BIGINT PRIMARY KEY
  is_fraud BOOLEAN
  -- ...
);