----- This file contains sample SQL queries to get common analytics from our OLAP database -----


-- Get all transactions pertaining to a given merchant

SELECT 
  t.transaction_id,
  t.time_id,
  t.customer_id,
  t.currency_code,
  t.ip_geo_id,
  t.payment_method_id,
  t.payment_status_id,
  t.device_type_id,
  t.amount
FROM
  transactions AS t
JOIN
  merchants
ON
  t.merchant_id = merchants.merchant_id
WHERE
  merchants.merchant_name = @merchant_name


-- Get quaterly transaction volume for each year in the USA

SELECT 
  TRUNC()
FROM
  transactions
JOIN
  merchants
ON
  transactions.merchant_id = merchants.merchant_id
WHERE
  merchants.merchant_name = @merchant_name



-- Get monthly transaction volume and transaction number for each currency


-- Get average amount per transaction yearly (expressed in USD)


-- Get average hourly transaction volume (with timezone adjustment)
