CREATE DATABASE stripe_olap

CREATE SCHEMA transactions AUTHORIZATION transactionsadmin;
GO

----- Main tables -----

CREATE TABLE transactions.transactions (
  transaction_id BIGINT PRIMARY KEY
  time_id BIGINT FOREIGN KEY
  merchant_id BIGINT FOREIGN KEY
  customer_id BIGINT FOREIGN KEY
  currency_code TEXT FOREIGN KEY -- ISO 4217
  ip_geo_id BIGINT FOREIGN KEY
  payment_method_id INTEGER FOREIGN KEY
  payment_status_id INTEGER FOREIGN KEY
  device_type_id INTEGER FOREIGN KEY
  amount DECIMAL(10, 2)
);

CREATE TABLE transactions.time {
  time_id BIGINT PRIMARY KEY
  timestamp DATETIME NOT NULL
  timezone INTEGER
  date DATE NOT NULL
  time TIME NOT NULL
  year MEDIUMINT NOT NULL
  month TINYINT FOREIGN KEY
  week_day TINYINT FOREIGN KEY
  quarter TINYINT NOT NULL
}

CREATE TABLE transactions.merchants (
  merchant_id BIGINT PRIMARY KEY
  merchant_name TEXT
  country_code TEXT FOREIGN KEY -- ISO 3166-1 alpha-2
  -- ...
);

CREATE TABLE transactions.customers (
  customer_id BIGINT PRIMARY KEY
  country_code TEXT FOREIGN KEY -- ISO 3166-1 alpha-2
  -- ...
);

CREATE TABLE transactions.fraud_indicators (
  transaction_id BIGINT FOREIGN KEY
  fraud_probability FLOAT
  -- ...
);

CREATE TABLE transactions.ip_geography {
  ip_geo_id BIGINT PRIMARY KEY
  latitude FLOAT
  longitude FLOAT
  country_code INTEGER FOREIGN KEY
  province TEXT
  -- ...
}


----- Reference tables -----

CREATE TABLE transactions.currencies {
  currency_code TEXT PRIMARY KEY -- ISO 4217
  currency_name TEXT NOT NULL
  usd_change_rate decimal(6,4)
}

CREATE TABLE transactions.countries {
  country_code TEXT PRIMARY KEY -- ISO 3166-1 alpha-2
  country_name TEXT NOT NULL
}

CREATE TABLE transactions.payment_methods_ref {
  payment_method_id INTEGER PRIMARY KEY
  payment_method TEXT NOT NULL -- credit_card, ...
}

CREATE TABLE transactions.payment_statuses_ref {
  payment_status_id INTEGER PRIMARY KEY
  payment_status TEXT NOT NULL -- successful, failed, refunded, ...
}

CREATE TABLE transactions.device_types_ref {
  device_type_id INTEGER PRIMARY KEY
  device_type TEXT NOT NULL -- mobile, desktop, ...
}

CREATE TABLE transactions.months_ref {
  month TINYINT PRIMARY KEY
  month_name TEXT NOT NULL
}

CREATE TABLE transactions.week_days_ref {
  week_day TINYINT PRIMARY KEY
  week_day_name TEXT NOT NULL
}
