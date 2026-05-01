# FastAPI Inference Demo Report
Generated: 2026-05-01 09:45:52 UTC
Model: tesco-customer-segmentation Production
Test customers: 200

## Score Distribution

| Metric | Value |
|--------|-------|
| MEAN   | 0.001 |
| STD    | 0.000 |
| MIN    | 0.000 |
| P10    | 0.000 |
| P25    | 0.000 |
| P50    | 0.001 |
| P75    | 0.001 |
| P90    | 0.001 |
| P95    | 0.001 |
| P99    | 0.001 |
| MAX    | 0.001 |

## Segment Distribution

| Segment | Count | % of customers |
|---------|-------|----------------|
| 0       | 114   | 57.0%           |
| 2       | 86    | 43.0%           |

## Top 10 Customers by Propensity Score

| customer_id | propensity_score | segment_id |
|-------------|-----------------|------------|
| CUST-00037  | 0.000917       | 0          |
| CUST-00010  | 0.000883       | 2          |
| CUST-00032  | 0.000873       | 0          |
| CUST-00117  | 0.000861       | 0          |
| CUST-00108  | 0.000860       | 0          |
| CUST-00172  | 0.000849       | 2          |
| CUST-00113  | 0.000848       | 2          |
| CUST-00105  | 0.000847       | 2          |
| CUST-00052  | 0.000831       | 0          |
| CUST-00114  | 0.000827       | 2          |

## Sample Explanations (Top 5 Customers)

### Customer ID: CUST-00010
**Propensity Score:** 0.001
**Explanation:** This customer received a propensity score of 0.00 primarily because their total spend (£326.81) is above average and they shopped recently (0 days ago).

**Top Features:**
  1. monetary: impact 0.0071 (positive)
  2. recency_days: impact 0.0000 (negative)
  3. frequency: impact 0.0000 (positive)
  4. avg_basket_size: impact 0.0000 (positive)
  5. basket_std: impact 0.0000 (positive)

### Customer ID: CUST-00032
**Propensity Score:** 0.001
**Explanation:** This customer received a propensity score of 0.00 primarily because their total spend (£330.85) is above average and their last purchase was 1 days ago, above the average.

**Top Features:**
  1. monetary: impact 0.0086 (positive)
  2. recency_days: impact 0.0000 (positive)
  3. frequency: impact 0.0000 (positive)
  4. avg_basket_size: impact 0.0000 (positive)
  5. basket_std: impact 0.0000 (positive)

### Customer ID: CUST-00037
**Propensity Score:** 0.001
**Explanation:** This customer received a propensity score of 0.00 primarily because their total spend (£314.92) is above average and they shopped recently (0 days ago).

**Top Features:**
  1. monetary: impact 0.0086 (positive)
  2. recency_days: impact 0.0000 (negative)
  3. frequency: impact 0.0000 (positive)
  4. avg_basket_size: impact 0.0000 (positive)
  5. basket_std: impact 0.0000 (positive)

### Customer ID: CUST-00108
**Propensity Score:** 0.001
**Explanation:** This customer received a propensity score of 0.00 primarily because their total spend (£335.93) is above average and their last purchase was 1 days ago, above the average.

**Top Features:**
  1. monetary: impact 0.0086 (positive)
  2. recency_days: impact 0.0000 (positive)
  3. frequency: impact 0.0000 (positive)
  4. avg_basket_size: impact 0.0000 (positive)
  5. basket_std: impact 0.0000 (positive)

### Customer ID: CUST-00117
**Propensity Score:** 0.001
**Explanation:** This customer received a propensity score of 0.00 primarily because their total spend (£335.53) is above average and their last purchase was 2 days ago, above the average.

**Top Features:**
  1. monetary: impact 0.0084 (positive)
  2. recency_days: impact 0.0000 (positive)
  3. frequency: impact 0.0000 (positive)
  4. avg_basket_size: impact 0.0000 (positive)
  5. basket_std: impact 0.0000 (positive)
