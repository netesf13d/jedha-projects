CREATE DATABASE stripe_olap

CREATE SCHEMA transactions AUTHORIZATION transactionsadmin;
GO

----- Main tables -----

CREATE TABLE transactions.transactions (
  transaction_id BIGINT PRIMARY KEY
  timestamp DATETIME FOREIGN KEY
  merchant_id BIGINT FOREIGN KEY
  customer_id BIGINT FOREIGN KEY
  currency_code CHAR(3) FOREIGN KEY -- ISO 4217
  ip_geo_id BIGINT FOREIGN KEY
  amount FLOAT
  fee FLOAT
  payment_method_id INTEGER FOREIGN KEY
  payment_status_id INTEGER FOREIGN KEY
  device_type_id INTEGER FOREIGN KEY
);

CREATE TABLE transactions.times {
  timestamp DATETIME PRIMARY KEY
  timezone INTEGER
  date DATE NOT NULL
  time TIME NOT NULL
  year INTEGER NOT NULL
  month TINYINT FOREIGN KEY
  week_day TINYINT FOREIGN KEY
  quarter TINYINT NOT NULL
}

CREATE TABLE transactions.merchants (
  merchant_id BIGINT PRIMARY KEY
  merchant_name TEXT
  country_code CHAR(2) FOREIGN KEY -- ISO 3166-1 alpha-2
  -- ...
);

CREATE TABLE transactions.customers (
  customer_id BIGINT PRIMARY KEY
  country_code TEXT FOREIGN KEY -- ISO 3166-1 alpha-2
  -- ...
);

CREATE TABLE transactions.fraud_indicators (
  transaction_id BIGINT PRIMARY KEY
  is_fraud BOOLEAN 
  fraud_probability FLOAT
  -- ...
);

CREATE TABLE transactions.ip_geography {
  ip_geo_id BIGINT PRIMARY KEY
  latitude FLOAT
  longitude FLOAT
  country_code CHAR(2) FOREIGN KEY
  province TEXT
  -- ...
}


----- Reference tables -----

CREATE TABLE transactions.currencies {
  currency_code CHAR(3) PRIMARY KEY -- ISO 4217
  currency_name TEXT NOT NULL
  usd_change_rate FLOAT
  -- ...
}

CREATE TABLE transactions.countries {
  country_code CHAR(2) PRIMARY KEY -- ISO 3166-1 alpha-2
  country_name TEXT NOT NULL
}

CREATE TABLE transactions.payment_methods {
  payment_method_id INTEGER PRIMARY KEY
  payment_method TEXT NOT NULL -- credit_card, ...
}

CREATE TABLE transactions.payment_statuses {
  payment_status_id INTEGER PRIMARY KEY
  payment_status TEXT NOT NULL -- successful, failed, refunded, ...
}

CREATE TABLE transactions.device_types {
  device_type_id INTEGER PRIMARY KEY
  device_type TEXT NOT NULL -- mobile, desktop, ...
}

CREATE TABLE transactions.months {
  month TINYINT PRIMARY KEY
  month_name TEXT NOT NULL
}

CREATE TABLE transactions.week_days {
  week_day TINYINT PRIMARY KEY
  week_day_name TEXT NOT NULL
}
