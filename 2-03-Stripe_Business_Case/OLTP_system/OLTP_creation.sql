CREATE DATABASE stripe_oltp

CREATE SCHEMA transactions AUTHORIZATION oltpadmin;
GO


CREATE TABLE transactions.transactions (
  transaction_id BIGINT PRIMARY KEY
  merchant_id BIGINT FOREIGN KEY
  customer_id BIGINT FOREIGN KEY
  timestamp DATETIME
  amount DECIMAL(10, 2)
  currency_code CHAR(3) -- ISO 4217
  payment_method VARCHAR(16)
  payment_status VARCHAR(16)
  device_type VARCHAR(16)
  ip_latitude FLOAT
  ip_longitude FLOAT
);

CREATE TABLE transactions.merchants (
  merchant_id BIGINT PRIMARY KEY
  name VARCHAR(255)
  iban VARCHAR(34)
  country_code CHAR(2) -- ISO 3166-1 alpha-2
  -- ...
);

CREATE TABLE transactions.customers (
  customer_id BIGINT PRIMARY KEY
  name VARCHAR(255)
  iban VARCHAR(34)
  country_code CHAR(2) -- ISO 3166-1 alpha-2
  -- ...
);

CREATE TABLE transactions.fraud_indicators (
  transaction_id BIGINT FOREIGN KEY
  fraud_probability FLOAT
  -- ...
);