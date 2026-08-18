# Lab 14 - Big Data Analytics with PySpark

## CST8921 - Cloud Industry Trends

This lab demonstrates big data analytics techniques using **Apache Spark and PySpark**. The lab works with a sample transaction dataset and applies descriptive analytics, diagnostic analytics, window functions, feature engineering, customer segmentation, anomaly detection, and Parquet-based data engineering.

The implementation was tested successfully using **PySpark 4.2.0**.

---

## Technologies Used

- Python
- Apache Spark
- PySpark 4.2.0
- Java 17
- WSL 2
- Ubuntu
- Parquet
- Git

---

## Project Structure

```text
lab14/
│
├── 00-setup.md
├── 01-descriptive-analytics.md
├── 02-diagnostic-analytics.md
├── 03-window-functions.md
├── 04-predictive-feature-engineering.md
├── 05-customer-segmentation.md
├── 06-anomaly-detection.md
├── 07-data-engineering-parquet.md
├── 08-hands-on-exercises.md
├── big_data_analytics_lab.py
├── lab14-final-output.txt
├── README.md
└── .gitignore
```

---

# Environment Setup

## 1. Windows Environment

The project files are stored in:

```text
C:\2026_2\8921_Industry_Trend\lab14
```

PySpark was initially tested on Windows. However, Hadoop file permission operations caused problems when writing Parquet files directly through the Windows-mounted filesystem.

For this reason, the complete lab is run using **Ubuntu through WSL 2**.

---

## 2. Start Ubuntu WSL

From Windows PowerShell:

```powershell
wsl -d Ubuntu
```

Navigate to the Lab 14 directory:

```bash
cd /mnt/c/2026_2/8921_Industry_Trend/lab14
```

---

## 3. Activate the Python Virtual Environment

The Linux virtual environment is stored in the Ubuntu home directory.

Activate it with:

```bash
source ~/lab14-venv/bin/activate
```

The command prompt should display:

```text
(lab14-venv)
```

Verify Python:

```bash
python --version
```

Verify PySpark:

```bash
python -c "import pyspark; print(pyspark.__version__)"
```

---

## 4. Java Requirement

Apache Spark requires Java.

This lab uses Java 17.

Verify Java with:

```bash
java -version
```

Example:

```text
openjdk version "17.0.19"
```

---

# Running the Lab

Run the complete PySpark program from WSL:

```bash
python big_data_analytics_lab.py
```

To save all program output to a file:

```bash
python big_data_analytics_lab.py > lab14-final-output.txt 2>&1
```

Check whether the program completed successfully:

```bash
echo $?
```

A result of:

```text
0
```

indicates successful program execution.

---

# Part 1 - Descriptive Analytics

The first part performs descriptive analysis of the transaction data.

The analysis includes:

- Summary statistics
- Revenue by category
- Revenue by region
- Average revenue
- Transaction counts
- Units sold

### Key Finding

Electronics generated the highest total revenue:

```text
Electronics: $6,179.47
Clothing:      $800.00
Food:          $739.50
```

North generated the highest regional revenue:

```text
North: $3,427.47
East:  $1,953.50
West:  $1,267.00
South: $1,071.00
```

---

# Part 2 - Diagnostic Analytics

Diagnostic analytics was used to investigate patterns in the transaction data.

Techniques included:

- Region/category pivot tables
- Regional drill-down
- Monthly revenue trends
- Payment method analysis

Credit-card transactions had the highest average transaction revenue.

```text
Credit Card: $684.77
Debit Card:  $157.30
Cash:        $128.25
```

---

# Part 3 - Window Functions

PySpark window functions were used to analyze transactions while preserving individual rows.

The analysis included:

- Revenue ranking within each region
- Top two transactions per region
- Running revenue totals
- Revenue quartiles
- Previous purchase per customer using `lag()`

Example:

```text
Alice
T001 -> First transaction
T004 -> Previous revenue: $1,799.98
T011 -> Previous revenue: $62.50
T020 -> Previous revenue: $250.00
```

Window functions allow calculations across related rows without collapsing the dataset through aggregation.

---

# Part 4 - Feature Engineering and RFM

Additional analytical features were created from the transaction data.

Examples include:

- Hour
- Day
- Day of week
- Month
- Weekend indicator
- High-value indicator
- High-quantity indicator

RFM analysis was also performed.

RFM represents:

- **R - Recency:** How recently the customer purchased
- **F - Frequency:** How often the customer purchased
- **M - Monetary:** How much the customer spent

Example RFM result:

```text
Alice
Recency = 0
Frequency = 4
Monetary = $3,112.47
RFM = R4F4M4
```

---

# Part 5 - Customer Segmentation

Customers were segmented according to their RFM scores.

Segments included:

- Champions
- Loyal
- Potential Loyalist
- New / Promising
- Hibernating / Lost

Example:

```text
Alice   -> Champions
David   -> Loyal
Heidi   -> Potential Loyalist
Grace   -> Potential Loyalist
Ivan    -> New / Promising
Bob     -> Hibernating / Lost
Frank   -> New / Promising
Eve     -> Hibernating / Lost
Charlie -> Hibernating / Lost
```

The segmentation can support different marketing strategies for different customer groups.

---

# Part 6 - Anomaly Detection

Z-scores were used to identify unusual transaction revenue.

The formula conceptually measures how far a transaction is from the mean in terms of standard deviations.

Using:

```text
|z| > 2
```

two global anomalies were detected:

```text
T001 - Alice - $1,799.98 - z = 2.843
T017 - Grace - $1,560.00 - z = 2.360
```

The effect of different thresholds was also tested:

```text
|z| > 1.5 -> 2 anomalies
|z| > 2.0 -> 2 anomalies
|z| > 2.5 -> 1 anomaly
|z| > 3.0 -> 0 anomalies
```

This demonstrates that higher thresholds make anomaly detection more restrictive.

---

# Part 7 - Data Engineering with Parquet

The enriched transaction dataset was written to **Parquet format** and read back into Spark.

Because Hadoop permission operations caused problems when writing to the Windows-mounted `/mnt/c` directory, the Parquet output was stored in the native WSL Linux filesystem:

```text
/home/systemadmin/lab14-output/transactions_parquet
```

The Parquet round-trip verification produced:

```text
Original row count: 20
Parquet row count:  20
Row count verification: PASSED
```

This confirmed that the dataset was successfully written and read without losing records.

---

# Part 8 - Hands-On Exercises

Several additional analytical exercises were completed.

## Exercise 1 - Most Expensive Category per Region

Average revenue per unit was calculated and categories were ranked within each region.

Electronics ranked first in all four regions.

```text
East  -> Electronics -> $489.75
North -> Electronics -> $716.66
South -> Electronics -> $320.00
West  -> Electronics -> $525.00
```

---

## Exercise 2 - Credit Card vs Cash

Average revenue was compared between credit-card and cash transactions.

```text
Credit Card Average: $684.77
Cash Average:        $128.25
Difference:          $556.52
```

Credit-card transactions had higher average revenue in this dataset.

---

## Exercise 3 - Weekend Effect

Weekend and weekday transactions were compared.

```text
Weekday Average Revenue: $400.31
Weekend Average Revenue: $359.29
Difference:              $41.02
```

Weekday transactions had higher average revenue.

---

## Exercise 4 - High Quantity vs Payment Method

A high-quantity transaction was defined as:

```text
quantity > 3
```

Results:

```text
Cash        -> 3 of 6 high quantity  -> 50.00%
Credit Card -> 1 of 9 high quantity  -> 11.11%
Debit Card  -> 0 of 5 high quantity  -> 0.00%
```

Cash transactions showed the highest proportion of high-quantity purchases in this sample.

Because the dataset contains only 20 transactions, this result should be interpreted as an observation rather than strong evidence of a general relationship.

---

## Exercise 5 - RFM Threshold Tuning

The RFM segmentation thresholds were modified to determine how sensitive the customer classifications were to the business rules.

Three customers changed segments:

```text
Bob:
Hibernating / Lost -> Loyal

Charlie:
Hibernating / Lost -> Loyal

Ivan:
New / Promising -> Potential Loyalist
```

This demonstrates that RFM segmentation depends on the thresholds selected by the business.

---

## Exercise 6 - Anomaly Threshold Comparison

The anomaly threshold was changed from:

```text
|z| > 2.0
```

to:

```text
|z| > 1.5
```

Both thresholds identified two anomalies.

Therefore, lowering the threshold did not identify additional anomalies because no transactions had absolute z-scores between 1.5 and 2.0.

---

# Final Challenge - Region Health Score

A composite region health score was created using:

- Total revenue
- Transaction activity
- Unique customers

Final results:

| Region | Total Revenue | Transactions | Unique Customers | Health Score | Status |
|---|---:|---:|---:|---:|---|
| North | $3,427.47 | 6 | 2 | 11 | Strong |
| South | $1,071.00 | 6 | 3 | 9 | Moderate |
| East | $1,953.50 | 4 | 2 | 9 | Moderate |
| West | $1,267.00 | 4 | 2 | 8 | Moderate |

North achieved the highest overall health score because it generated the highest revenue and tied for the highest transaction count.

The composite score demonstrates how multiple KPIs can be combined rather than evaluating regional performance using revenue alone.

---

# Key Learning Outcomes

This lab demonstrates how PySpark can be used for scalable analytical workloads.

The main concepts practiced were:

1. Spark DataFrames
2. Aggregations
3. Grouping and pivot operations
4. Window functions
5. Feature engineering
6. RFM analysis
7. Customer segmentation
8. Z-score anomaly detection
9. Parquet data storage
10. Business-oriented analytical interpretation

The lab also demonstrated the difference between performing analytics and turning analytical results into actionable business information.

---

# Environment Note

The main Python source code is stored on the Windows filesystem:

```text
C:\2026_2\8921_Industry_Trend\lab14
```

The program is executed through Ubuntu WSL using:

```text
/mnt/c/2026_2/8921_Industry_Trend/lab14
```

Parquet output is written to the native Linux filesystem to avoid Hadoop permission issues on the Windows-mounted filesystem:

```text
/home/systemadmin/lab14-output/transactions_parquet
```

For this project:

- **WSL/Ubuntu** is used to execute PySpark.
- **Windows PowerShell** should be used for Git operations on the Windows project directory.

---

# Final Verification

The complete application was executed using:

```bash
python big_data_analytics_lab.py > lab14-final-output.txt 2>&1
```

The exit status was:

```text
0
```

This confirms that the complete Lab 14 PySpark program executed successfully.