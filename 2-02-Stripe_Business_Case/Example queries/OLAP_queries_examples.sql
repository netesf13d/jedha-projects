----- This file contains sample SQL queries to get common analytics from the OLAP database -----


-- Top merchants by Stripe revenue in @year

DECLARE @year INTEGER = 2024;

SELECT
  merchant_id,
  merchants.merchant_name,
  merchants.country_code,
  nb_transactions,
  total_amount,
  total_revenue
FROM
  (
    SELECT 
      merchant_id,
      COUNT(merchant_id) AS nb_transactions,
      SUM(amount) AS total_amount,
      SUM(fee) AS total_revenue -- this is stripe actual revenue
    FROM
      transactions
    JOIN
      datetimes
    ON
      timestamp = datetimes.timestamp
    WHERE
      datetimes.year = @year
    GROUP BY
      merchant_id
  )
JOIN
  merchants
ON
  merchants.merchant_id == merchant_id
ORDER by
  total_revenue


-- Get monthly transaction info in @country

DECLARE @country TEXT = 'US';

SELECT 
  TRUNC(timestamp, 'month') as date,
  COUNT(amount) AS monthly_transaction_count,
  SUM(amount) AS monthly_amount,
  SUM(fee) AS monthly_revenue
FROM
  transactions
WHERE
  country_code = @country
GROUP BY
  TRUNC(timestamp, 'month')
ORDER BY
  date


-- Get average hourly transaction volume in @country

DECLARE @country TEXT = 'FR';

SELECT 
  hour,
  SUM(hourly_transactions) / COUNT(hourly_transactions) AS avg_hourly_transactions,
  SUM(hourly_amount) / COUNT(hourly_amount) AS avg_hourly_amount,
  SUM(hourly_revenue) / Count(hourly_revenue) AS avg_hourly_revenue
FROM
  (
    SELECT 
      HOUR(timestamp) as hour,
      COUNT(amount) AS hourly_transactions,
      SUM(amount) AS hourly_amount,
      SUM(fee) AS hourly_revenue
    FROM
      transactions
    WHERE
      country_code = @country
    GROUP BY
      DATE_TRUNC(HOUR, timestamp)
  )
GROUP BY
  hour
ORDER BY
  hour


-- monthly fraud rate in @country

DECLARE @country TEXT = 'GB';

SELECT 
  TRUNC(timestamp, 'month') as date,
  SUM(CAST(is_fraud AS FLOAT)) / COUNT(is_fraud) AS fraud_rate,
FROM
  transactions
WHERE
  country_code = @country
GROUP BY
  TRUNC(timestamp, 'month')
ORDER BY
  date