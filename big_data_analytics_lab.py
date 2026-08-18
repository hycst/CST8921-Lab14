import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession


from pyspark.sql.window import Window

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DoubleType,
)
from pyspark.sql import functions as F


# ============================================================
# Lab 14 - Big Data Analytics with PySpark
# Part 0: Setup
# ============================================================

# Step 1: Create SparkSession



spark = (
    SparkSession.builder
    .appName("Lab14-Big-Data-Analytics")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "2")
    .config(
        "spark.hadoop.fs.file.impl",
        "org.apache.hadoop.fs.RawLocalFileSystem"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


spark.sparkContext.setLogLevel("ERROR")

print("=" * 60)
print("LAB 14 - BIG DATA ANALYTICS WITH PYSPARK")
print("=" * 60)

print("\nSpark Version:", spark.version)
print("Spark Session created successfully!")



# ============================================================
# Step 2: Define the transaction dataset
# ============================================================

transactions = [
    (1,  "T001", "Alice",   "North", "Electronics", 899.99, 2, "2024-01-05 10:30:00", "credit_card"),
    (2,  "T002", "Bob",     "South", "Clothing",     45.00, 3, "2024-01-06 11:00:00", "cash"),
    (3,  "T003", "Charlie", "East",  "Electronics", 199.50, 1, "2024-01-06 14:20:00", "debit_card"),
    (4,  "T004", "Alice",   "North", "Food",          12.50, 5, "2024-01-07 09:15:00", "cash"),
    (5,  "T005", "David",   "West",  "Electronics", 450.00, 1, "2024-01-08 16:45:00", "credit_card"),
    (6,  "T006", "Eve",     "South", "Food",          22.00, 4, "2024-01-08 18:00:00", "credit_card"),
    (7,  "T007", "Frank",   "North", "Clothing",      75.00, 2, "2024-01-09 13:30:00", "debit_card"),
    (8,  "T008", "Grace",   "East",  "Food",          33.00, 3, "2024-01-10 10:00:00", "cash"),
    (9,  "T009", "Heidi",   "West",  "Electronics", 600.00, 1, "2024-02-01 12:00:00", "credit_card"),
    (10, "T010", "Ivan",    "South", "Clothing",     110.00, 2, "2024-02-02 15:30:00", "debit_card"),
    (11, "T011", "Alice",   "North", "Electronics", 250.00, 1, "2024-02-03 09:00:00", "credit_card"),
    (12, "T012", "Bob",     "South", "Food",          18.00, 6, "2024-02-04 17:00:00", "cash"),
    (13, "T013", "Charlie", "East",  "Clothing",      95.00, 1, "2024-02-05 11:45:00", "credit_card"),
    (14, "T014", "David",   "West",  "Food",           8.50, 2, "2024-02-06 08:30:00", "debit_card"),
    (15, "T015", "Eve",     "South", "Electronics", 320.00, 1, "2024-02-07 14:00:00", "credit_card"),
    (16, "T016", "Frank",   "North", "Food",          55.00, 3, "2024-03-01 10:15:00", "cash"),
    (17, "T017", "Grace",   "East",  "Electronics", 780.00, 2, "2024-03-02 16:00:00", "credit_card"),
    (18, "T018", "Heidi",   "West",  "Clothing",     200.00, 1, "2024-03-03 12:30:00", "debit_card"),
    (19, "T019", "Ivan",    "South", "Food",          40.00, 5, "2024-03-04 09:45:00", "cash"),
    (20, "T020", "Alice",   "North", "Electronics", 999.99, 1, "2024-03-05 11:00:00", "credit_card"),
]


# ============================================================
# Step 3: Define explicit schema
# ============================================================

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("transaction_id", StringType(), True),
    StructField("customer", StringType(), True),
    StructField("region", StringType(), True),
    StructField("category", StringType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("timestamp", StringType(), True),
    StructField("payment_method", StringType(), True),
])


# Create DataFrame
df = spark.createDataFrame(transactions, schema=schema)

print("\n" + "=" * 60)
print("ORIGINAL TRANSACTION DATA")
print("=" * 60)

df.show(20, truncate=False)

print("\nNumber of transactions:", df.count())

print("\nSchema:")
df.printSchema()



# ============================================================
# Step 4: Add derived columns
# ============================================================

df = (
    df
    .withColumn(
        "revenue",
        F.round(F.col("unit_price") * F.col("quantity"), 2)
    )
    .withColumn(
        "event_time",
        F.to_timestamp(
            F.col("timestamp"),
            "yyyy-MM-dd HH:mm:ss"
        )
    )
)

# Cache because this DataFrame will be reused throughout the lab
df.cache()

print("\n" + "=" * 60)
print("ENRICHED TRANSACTION DATA")
print("=" * 60)

df.select(
    "transaction_id",
    "customer",
    "region",
    "category",
    "unit_price",
    "quantity",
    "revenue",
    "event_time"
).show(20, truncate=False)

print("\nEnriched Schema:")
df.printSchema()




# ============================================================
# Part 1: Descriptive Analytics
# Step 5: Summary Statistics
# ============================================================

print("\n" + "=" * 60)
print("PART 1 - DESCRIPTIVE ANALYTICS")
print("=" * 60)

print("\n1. SUMMARY STATISTICS")
df.select(
    "unit_price",
    "quantity",
    "revenue"
).describe().show()






# Revenue by category
print("\n2. REVENUE BY CATEGORY")

category_summary = (
    df.groupBy("category")
    .agg(
        F.round(F.sum("revenue"), 2).alias("total_revenue"),
        F.round(F.avg("revenue"), 2).alias("avg_revenue"),
        F.count("*").alias("transaction_count"),
        F.sum("quantity").alias("units_sold")
    )
    .orderBy(F.desc("total_revenue"))
)

category_summary.show()


# Revenue by region
print("\n3. REVENUE BY REGION")

region_summary = (
    df.groupBy("region")
    .agg(
        F.round(F.sum("revenue"), 2).alias("total_revenue"),
        F.round(F.avg("revenue"), 2).alias("avg_revenue"),
        F.count("*").alias("transaction_count"),
        F.sum("quantity").alias("units_sold")
    )
    .orderBy(F.desc("total_revenue"))
)

region_summary.show()




# ============================================================
# Part 2: Diagnostic Analytics
# ============================================================

print("\n" + "=" * 60)
print("PART 2 - DIAGNOSTIC ANALYTICS")
print("=" * 60)


# ------------------------------------------------------------
# 1. Region x Category Revenue Pivot
# ------------------------------------------------------------

print("\n1. REGION x CATEGORY REVENUE PIVOT")

region_category_pivot = (
    df.groupBy("region")
    .pivot("category")
    .agg(F.round(F.sum("revenue"), 2))
    .orderBy("region")
)

region_category_pivot.show()





# ------------------------------------------------------------
# 2. Drill-down into top revenue region: North
# ------------------------------------------------------------

print("\n2. DRILL-DOWN - NORTH REGION")

north_transactions = (
    df.filter(F.col("region") == "North")
    .select(
        "transaction_id",
        "customer",
        "category",
        "revenue",
        "event_time"
    )
    .orderBy(F.desc("revenue"))
)

north_transactions.show(truncate=False)




# ------------------------------------------------------------
# 3. Monthly Revenue Trend
# ------------------------------------------------------------

print("\n3. MONTHLY REVENUE TREND")

monthly_trend = (
    df.withColumn(
        "month",
        F.date_format("event_time", "yyyy-MM")
    )
    .groupBy("month")
    .agg(
        F.round(F.sum("revenue"), 2).alias("total_revenue"),
        F.count("*").alias("transaction_count")
    )
    .orderBy("month")
)

monthly_trend.show()



# ------------------------------------------------------------
# 4. Average Revenue by Payment Method
# ------------------------------------------------------------

print("\n4. AVERAGE REVENUE BY PAYMENT METHOD")

payment_summary = (
    df.groupBy("payment_method")
    .agg(
        F.round(F.avg("revenue"), 2).alias("avg_revenue"),
        F.round(F.sum("revenue"), 2).alias("total_revenue"),
        F.count("*").alias("transaction_count")
    )
    .orderBy(F.desc("avg_revenue"))
)

payment_summary.show()





# ============================================================
# Part 3: Window Functions
# ============================================================

print("\n" + "=" * 60)
print("PART 3 - WINDOW FUNCTIONS")
print("=" * 60)


# ------------------------------------------------------------
# 1. Rank transactions within each region
# ------------------------------------------------------------

print("\n1. TRANSACTION RANK WITHIN EACH REGION")

region_rank_window = (
    Window
    .partitionBy("region")
    .orderBy(F.desc("revenue"))
)

ranked_df = df.withColumn(
    "revenue_rank",
    F.rank().over(region_rank_window)
)

ranked_df.select(
    "region",
    "transaction_id",
    "customer",
    "revenue",
    "revenue_rank"
).orderBy("region", "revenue_rank").show(20, truncate=False)



# ------------------------------------------------------------
# 2. Top 2 transactions per region
# ------------------------------------------------------------

print("\n2. TOP 2 TRANSACTIONS PER REGION")

top_2_by_region = (
    ranked_df
    .filter(F.col("revenue_rank") <= 2)
    .select(
        "region",
        "transaction_id",
        "customer",
        "revenue",
        "revenue_rank"
    )
    .orderBy("region", "revenue_rank")
)

top_2_by_region.show(truncate=False)



# ------------------------------------------------------------
# 3. Running revenue total by region
# ------------------------------------------------------------

print("\n3. RUNNING REVENUE TOTAL BY REGION")

running_window = (
    Window
    .partitionBy("region")
    .orderBy("event_time")
    .rowsBetween(
        Window.unboundedPreceding,
        Window.currentRow
    )
)

running_total_df = df.withColumn(
    "running_revenue",
    F.round(
        F.sum("revenue").over(running_window),
        2
    )
)

running_total_df.select(
    "region",
    "transaction_id",
    "event_time",
    "revenue",
    "running_revenue"
).orderBy("region", "event_time").show(20, truncate=False)


# ------------------------------------------------------------
# 4. Revenue quartiles
# ------------------------------------------------------------

print("\n4. REVENUE QUARTILES")

quartile_window = Window.orderBy("revenue")

quartile_df = df.withColumn(
    "revenue_quartile",
    F.ntile(4).over(quartile_window)
)

quartile_df.select(
    "transaction_id",
    "customer",
    "revenue",
    "revenue_quartile"
).orderBy("revenue").show(20, truncate=False)



# ------------------------------------------------------------
# 5. Previous purchase per customer
# ------------------------------------------------------------

print("\n5. PREVIOUS PURCHASE PER CUSTOMER")

customer_window = (
    Window
    .partitionBy("customer")
    .orderBy("event_time")
)

previous_purchase_df = df.withColumn(
    "prev_revenue",
    F.lag("revenue").over(customer_window)
)

previous_purchase_df.select(
    "customer",
    "transaction_id",
    "event_time",
    "revenue",
    "prev_revenue"
).orderBy("customer", "event_time").show(20, truncate=False)





# ============================================================
# Part 4: Feature Engineering and RFM
# ============================================================

print("\n" + "=" * 60)
print("PART 4 - FEATURE ENGINEERING AND RFM")
print("=" * 60)


# ------------------------------------------------------------
# 1. Transaction-level time features
# ------------------------------------------------------------

print("\n1. TRANSACTION-LEVEL FEATURES")

avg_revenue = df.agg(
    F.avg("revenue").alias("avg_revenue")
).first()["avg_revenue"]

feature_df = (
    df
    .withColumn("hour", F.hour("event_time"))
    .withColumn("day", F.dayofmonth("event_time"))
    .withColumn("dayofweek", F.dayofweek("event_time"))
    .withColumn("month", F.month("event_time"))
    .withColumn(
        "is_weekend",
        F.when(
            F.col("dayofweek").isin(1, 7),
            1
        ).otherwise(0)
    )
    .withColumn(
        "is_high_value",
        F.when(
            F.col("revenue") > avg_revenue,
            1
        ).otherwise(0)
    )
    .withColumn(
        "high_quantity",
        F.when(
            F.col("quantity") > 3,
            1
        ).otherwise(0)
    )
)

feature_df.select(
    "transaction_id",
    "event_time",
    "revenue",
    "quantity",
    "hour",
    "day",
    "dayofweek",
    "month",
    "is_weekend",
    "is_high_value",
    "high_quantity"
).show(20, truncate=False)




# ------------------------------------------------------------
# 2. RFM reference date
# ------------------------------------------------------------

print("\n2. RFM REFERENCE DATE")

reference_date = df.agg(
    F.max("event_time").alias("max_date")
).first()["max_date"]

print("Reference date:", reference_date)





# ------------------------------------------------------------
# 3. Customer-level RFM
# ------------------------------------------------------------

print("\n3. RAW CUSTOMER RFM VALUES")

rfm_df = (
    feature_df
    .groupBy("customer")
    .agg(
        F.datediff(
            F.lit(reference_date),
            F.max("event_time")
        ).alias("recency_days"),

        F.count("*").alias("frequency"),

        F.round(
            F.sum("revenue"),
            2
        ).alias("monetary")
    )
)

rfm_df.orderBy("customer").show(truncate=False)



# ------------------------------------------------------------
# 4. Score RFM into tiers 1-4
# ------------------------------------------------------------

print("\n4. RFM SCORES")

recency_window = Window.orderBy(F.col("recency_days").asc())

frequency_window = Window.orderBy(F.col("frequency").asc())

monetary_window = Window.orderBy(F.col("monetary").asc())

rfm_scored_df = (
    rfm_df
    .withColumn(
        "R",
        5 - F.ntile(4).over(recency_window)
    )
    .withColumn(
        "F",
        F.ntile(4).over(frequency_window)
    )
    .withColumn(
        "M",
        F.ntile(4).over(monetary_window)
    )
    .withColumn(
        "rfm_sum",
        F.col("R") + F.col("F") + F.col("M")
    )
    .withColumn(
        "rfm_cell",
        F.concat(
            F.lit("R"),
            F.col("R"),
            F.lit("F"),
            F.col("F"),
            F.lit("M"),
            F.col("M")
        )
    )
)

rfm_scored_df.select(
    "customer",
    "recency_days",
    "frequency",
    "monetary",
    "R",
    "F",
    "M",
    "rfm_sum",
    "rfm_cell"
).orderBy(
    F.desc("rfm_sum")
).show(truncate=False)



# ============================================================
# Part 5: Customer Segmentation
# ============================================================

print("\n" + "=" * 60)
print("PART 5 - CUSTOMER SEGMENTATION")
print("=" * 60)


# ------------------------------------------------------------
# 1. Calculate FM score
# ------------------------------------------------------------

segmented_df = rfm_scored_df.withColumn(
    "fm",
    (F.col("F") + F.col("M")) / 2
)


# ------------------------------------------------------------
# 2. Assign customer segments
# ------------------------------------------------------------

segmented_df = segmented_df.withColumn(
    "segment",

    F.when(
        (F.col("R") >= 4) & (F.col("fm") >= 4),
        "Champions"
    )

    .when(
        F.col("fm") >= 3,
        "Loyal"
    )

    .when(
        (F.col("R") >= 3) & (F.col("fm") >= 2),
        "Potential Loyalist"
    )

    .when(
        F.col("R") >= 3,
        "New / Promising"
    )

    .when(
        (F.col("R") <= 2) & (F.col("fm") >= 3),
        "At Risk"
    )

    .when(
        (F.col("R") <= 2) & (F.col("fm") < 3),
        "Hibernating / Lost"
    )

    .otherwise("Needs Attention")
)


print("\n1. CUSTOMER SEGMENTS")

segmented_df.select(
    "customer",
    "recency_days",
    "frequency",
    "monetary",
    "R",
    "F",
    "M",
    "fm",
    "rfm_sum",
    "segment"
).orderBy(
    F.desc("rfm_sum")
).show(truncate=False)





# ============================================================
# Part 5: Customer Segmentation
# ============================================================

print("\n" + "=" * 60)
print("PART 5 - CUSTOMER SEGMENTATION")
print("=" * 60)


# ------------------------------------------------------------
# 1. Calculate FM score
# ------------------------------------------------------------

segmented_df = rfm_scored_df.withColumn(
    "fm",
    (F.col("F") + F.col("M")) / 2
)


# ------------------------------------------------------------
# 2. Assign customer segments
# ------------------------------------------------------------

segmented_df = segmented_df.withColumn(
    "segment",

    F.when(
        (F.col("R") >= 4) & (F.col("fm") >= 4),
        "Champions"
    )

    .when(
        F.col("fm") >= 3,
        "Loyal"
    )

    .when(
        (F.col("R") >= 3) & (F.col("fm") >= 2),
        "Potential Loyalist"
    )

    .when(
        F.col("R") >= 3,
        "New / Promising"
    )

    .when(
        (F.col("R") <= 2) & (F.col("fm") >= 3),
        "At Risk"
    )

    .when(
        (F.col("R") <= 2) & (F.col("fm") < 3),
        "Hibernating / Lost"
    )

    .otherwise("Needs Attention")
)


print("\n1. CUSTOMER SEGMENTS")

segmented_df.select(
    "customer",
    "recency_days",
    "frequency",
    "monetary",
    "R",
    "F",
    "M",
    "fm",
    "rfm_sum",
    "segment"
).orderBy(
    F.desc("rfm_sum")
).show(truncate=False)



# ============================================================
# Part 6: Anomaly Detection
# ============================================================

print("\n" + "=" * 60)
print("PART 6 - ANOMALY DETECTION")
print("=" * 60)


# ------------------------------------------------------------
# 1. Calculate global revenue statistics
# ------------------------------------------------------------

print("\n1. GLOBAL REVENUE STATISTICS")

revenue_stats = df.agg(
    F.mean("revenue").alias("mean_revenue"),
    F.stddev("revenue").alias("stddev_revenue")
).first()

mean_revenue = revenue_stats["mean_revenue"]
stddev_revenue = revenue_stats["stddev_revenue"]

print(f"Mean Revenue:  {mean_revenue:.2f}")
print(f"Std Deviation: {stddev_revenue:.2f}")


# ------------------------------------------------------------
# 2. Calculate z-score for every transaction
# ------------------------------------------------------------

print("\n2. TRANSACTIONS WITH Z-SCORES")

THRESHOLD = 2.0

anomaly_df = (
    df
    .withColumn(
        "z",
        (F.col("revenue") - F.lit(mean_revenue))
        / F.lit(stddev_revenue)
    )
    .withColumn(
        "is_anomaly",
        F.abs(F.col("z")) > THRESHOLD
    )
)

anomaly_df.select(
    "transaction_id",
    "customer",
    "category",
    "revenue",
    F.round("z", 3).alias("z"),
    "is_anomaly"
).orderBy(
    F.desc(F.abs(F.col("z")))
).show(20, truncate=False)


# ------------------------------------------------------------
# 3. Show global anomalies
# ------------------------------------------------------------

print("\n3. GLOBAL ANOMALIES - |z| > 2")

global_anomalies = (
    anomaly_df
    .filter(F.col("is_anomaly") == True)
    .select(
        "transaction_id",
        "customer",
        "category",
        "revenue",
        F.round("z", 3).alias("z")
    )
    .orderBy(F.desc("revenue"))
)

global_anomalies.show(truncate=False)

print(
    "Number of global anomalies:",
    anomaly_df.filter(F.col("is_anomaly") == True).count()
)


# ------------------------------------------------------------
# 4. Per-category anomaly detection
# ------------------------------------------------------------

print("\n4. PER-CATEGORY ANOMALY DETECTION")

category_window = Window.partitionBy("category")

category_anomaly_df = (
    df
    .withColumn(
        "category_mean",
        F.mean("revenue").over(category_window)
    )
    .withColumn(
        "category_stddev",
        F.stddev("revenue").over(category_window)
    )
    .withColumn(
        "category_z",
        (F.col("revenue") - F.col("category_mean"))
        / F.col("category_stddev")
    )
    .withColumn(
        "category_anomaly",
        F.abs(F.col("category_z")) > THRESHOLD
    )
)

category_anomaly_df.select(
    "transaction_id",
    "customer",
    "category",
    "revenue",
    F.round("category_mean", 2).alias("category_mean"),
    F.round("category_stddev", 2).alias("category_stddev"),
    F.round("category_z", 3).alias("category_z"),
    "category_anomaly"
).orderBy(
    "category",
    F.desc(F.abs(F.col("category_z")))
).show(20, truncate=False)





# ============================================================
# Part 7: Data Engineering with Parquet
# ============================================================

print("\n" + "=" * 60)
print("PART 7 - DATA ENGINEERING WITH PARQUET")
print("=" * 60)


# ------------------------------------------------------------
# 1. Write enriched DataFrame to partitioned Parquet
# ------------------------------------------------------------

# output_path = "output/transactions_parquet"

output_path = "/home/systemadmin/lab14-output/transactions_parquet"

print("\n1. WRITING PARQUET DATA")
print("Output path:", output_path)

(
    feature_df
    .write
    .mode("overwrite")
    .partitionBy("region")
    .parquet(output_path)
)

print("Parquet write completed successfully!")


# ------------------------------------------------------------
# 2. Read Parquet data back
# ------------------------------------------------------------

print("\n2. READING PARQUET DATA")

parquet_df = spark.read.parquet(output_path)

print("Parquet data read successfully!")


# ------------------------------------------------------------
# 3. Verify round trip
# ------------------------------------------------------------

print("\n3. VERIFYING PARQUET ROUND TRIP")

original_count = feature_df.count()
parquet_count = parquet_df.count()

print("Original row count:", original_count)
print("Parquet row count: ", parquet_count)

if original_count == parquet_count:
    print("Row count verification: PASSED")
else:
    print("Row count verification: FAILED")


print("\nParquet Schema:")
parquet_df.printSchema()


print("\nSample Parquet Data:")

parquet_df.select(
    "transaction_id",
    "customer",
    "category",
    "revenue",
    "event_time",
    "region"
).orderBy(
    "region",
    "transaction_id"
).show(20, truncate=False)








# ============================================================
# Part 8: Hands-On Exercises
# Exercise 1 - Most Expensive Category per Region
# ============================================================

print("\n" + "=" * 60)
print("PART 8 - HANDS-ON EXERCISES")
print("=" * 60)

print("\nEXERCISE 1 - MOST EXPENSIVE CATEGORY PER REGION")


# ------------------------------------------------------------
# Step 1: Add revenue_per_unit
# ------------------------------------------------------------

exercise1_df = df.withColumn(
    "revenue_per_unit",
    F.round(
        F.col("revenue") / F.col("quantity"),
        2
    )
)

exercise1_df.select(
    "transaction_id",
    "region",
    "category",
    "revenue",
    "quantity",
    "revenue_per_unit"
).show(20, truncate=False)




# ------------------------------------------------------------
# Step 2: Average revenue per unit by region and category
# ------------------------------------------------------------

region_category_price = (
    exercise1_df
    .groupBy("region", "category")
    .agg(
        F.round(
            F.avg("revenue_per_unit"),
            2
        ).alias("avg_revenue_per_unit")
    )
)

print("\nAVERAGE REVENUE PER UNIT BY REGION AND CATEGORY")

region_category_price.orderBy(
    "region",
    F.desc("avg_revenue_per_unit")
).show(truncate=False)




# ------------------------------------------------------------
# Step 3: Rank categories within each region
# ------------------------------------------------------------

region_category_window = (
    Window
    .partitionBy("region")
    .orderBy(F.desc("avg_revenue_per_unit"))
)

ranked_categories = (
    region_category_price
    .withColumn(
        "category_rank",
        F.rank().over(region_category_window)
    )
)

print("\nCATEGORY RANK WITHIN EACH REGION")

ranked_categories.orderBy(
    "region",
    "category_rank"
).show(truncate=False)




# ------------------------------------------------------------
# Step 4: Most expensive category per region
# ------------------------------------------------------------

print("\nMOST EXPENSIVE CATEGORY PER REGION")

top_category_per_region = (
    ranked_categories
    .filter(F.col("category_rank") == 1)
    .select(
        "region",
        "category",
        "avg_revenue_per_unit"
    )
    .orderBy("region")
)

top_category_per_region.show(truncate=False)






# ============================================================
# Exercise 2 - Credit Card vs Cash
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 2 - CREDIT CARD VS CASH")
print("=" * 60)


# ------------------------------------------------------------
# Step 1: Keep only credit card and cash transactions
# ------------------------------------------------------------

payment_comparison_df = (
    df
    .filter(
        F.col("payment_method").isin("credit_card", "cash")
    )
)


# ------------------------------------------------------------
# Step 2: Compare average revenue
# ------------------------------------------------------------

payment_summary = (
    payment_comparison_df
    .groupBy("payment_method")
    .agg(
        F.round(F.avg("revenue"), 2).alias("avg_revenue"),
        F.round(F.sum("revenue"), 2).alias("total_revenue"),
        F.count("*").alias("transaction_count")
    )
    .orderBy(F.desc("avg_revenue"))
)

print("\nPAYMENT METHOD COMPARISON")

payment_summary.show(truncate=False)





payment_values = {
    row["payment_method"]: row["avg_revenue"]
    for row in payment_summary.collect()
}

credit_avg = payment_values.get("credit_card", 0)
cash_avg = payment_values.get("cash", 0)

difference = credit_avg - cash_avg

print(f"\nCredit Card Average Revenue: ${credit_avg:.2f}")
print(f"Cash Average Revenue:        ${cash_avg:.2f}")
print(f"Difference:                  ${difference:.2f}")

if credit_avg > cash_avg:
    print("Conclusion: Credit card transactions have higher average revenue.")
elif credit_avg < cash_avg:
    print("Conclusion: Cash transactions have higher average revenue.")
else:
    print("Conclusion: Both payment methods have the same average revenue.")




    # ============================================================
# Exercise 3 - Weekend Effect
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 3 - WEEKEND EFFECT")
print("=" * 60)


# ------------------------------------------------------------
# Step 1: Create readable day type
# ------------------------------------------------------------

weekend_df = (
    feature_df
    .withColumn(
        "day_type",
        F.when(
            F.col("is_weekend") == 1,
            "Weekend"
        ).otherwise("Weekday")
    )
)


# ------------------------------------------------------------
# Step 2: Compare weekend vs weekday
# ------------------------------------------------------------

weekend_summary = (
    weekend_df
    .groupBy("day_type")
    .agg(
        F.round(F.avg("revenue"), 2).alias("avg_revenue"),
        F.round(F.sum("revenue"), 2).alias("total_revenue"),
        F.count("*").alias("transaction_count"),
        F.sum("quantity").alias("units_sold")
    )
    .orderBy(F.desc("avg_revenue"))
)

print("\nWEEKEND VS WEEKDAY COMPARISON")

weekend_summary.show(truncate=False)




# ------------------------------------------------------------
# Step 3: Determine which has higher average revenue
# ------------------------------------------------------------

day_values = {
    row["day_type"]: row["avg_revenue"]
    for row in weekend_summary.collect()
}

weekend_avg = day_values.get("Weekend", 0)
weekday_avg = day_values.get("Weekday", 0)

difference = weekend_avg - weekday_avg

print(f"\nWeekend Average Revenue: ${weekend_avg:.2f}")
print(f"Weekday Average Revenue: ${weekday_avg:.2f}")
print(f"Difference:              ${difference:.2f}")

if weekend_avg > weekday_avg:
    print("Conclusion: Weekend transactions have higher average revenue.")
elif weekend_avg < weekday_avg:
    print("Conclusion: Weekday transactions have higher average revenue.")
else:
    print("Conclusion: Weekend and weekday transactions have the same average revenue.")





# ============================================================
# Exercise 4 - Customer Purchase Change
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 4 - CUSTOMER PURCHASE CHANGE")
print("=" * 60)


# Window: transactions ordered chronologically for each customer
customer_window = (
    Window
    .partitionBy("customer")
    .orderBy("event_time")
)


# ------------------------------------------------------------
# Step 1: Previous purchase
# ------------------------------------------------------------

purchase_change_df = (
    feature_df
    .withColumn(
        "previous_revenue",
        F.lag("revenue", 1).over(customer_window)
    )
)


# ------------------------------------------------------------
# Step 2: Calculate revenue change
# ------------------------------------------------------------

purchase_change_df = (
    purchase_change_df
    .withColumn(
        "revenue_change",
        F.round(
            F.col("revenue") - F.col("previous_revenue"),
            2
        )
    )
)


# ------------------------------------------------------------
# Step 3: Classify the change
# ------------------------------------------------------------

purchase_change_df = (
    purchase_change_df
    .withColumn(
        "purchase_trend",
        F.when(
            F.col("previous_revenue").isNull(),
            "First Purchase"
        )
        .when(
            F.col("revenue_change") > 0,
            "Increase"
        )
        .when(
            F.col("revenue_change") < 0,
            "Decrease"
        )
        .otherwise("No Change")
    )
)


print("\nCUSTOMER PURCHASE CHANGES")

purchase_change_df.select(
    "customer",
    "transaction_id",
    "event_time",
    "revenue",
    "previous_revenue",
    "revenue_change",
    "purchase_trend"
).orderBy(
    "customer",
    "event_time"
).show(50, truncate=False)






# ============================================================
# Exercise 5 - High-Value Customers
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 5 - HIGH-VALUE CUSTOMERS")
print("=" * 60)


# ------------------------------------------------------------
# Step 1: Calculate total revenue per customer
# ------------------------------------------------------------

customer_value_df = (
    feature_df
    .groupBy("customer")
    .agg(
        F.round(F.sum("revenue"), 2).alias("total_spending"),
        F.count("*").alias("transaction_count"),
        F.round(F.avg("revenue"), 2).alias("avg_transaction")
    )
)

print("\nCUSTOMER TOTAL SPENDING")

customer_value_df.orderBy(
    F.desc("total_spending")
).show(truncate=False)


# ------------------------------------------------------------
# Step 2: Calculate average customer spending
# ------------------------------------------------------------

avg_customer_spending = (
    customer_value_df
    .agg(F.avg("total_spending").alias("average"))
    .first()["average"]
)

print(
    f"\nAverage Customer Spending: "
    f"${avg_customer_spending:.2f}"
)


# ------------------------------------------------------------
# Step 3: Identify high-value customers
# ------------------------------------------------------------

high_value_customers = (
    customer_value_df
    .withColumn(
        "customer_type",
        F.when(
            F.col("total_spending") > avg_customer_spending,
            "High Value"
        ).otherwise("Regular")
    )
)

print("\nCUSTOMER VALUE CLASSIFICATION")

high_value_customers.orderBy(
    F.desc("total_spending")
).show(truncate=False)


# ------------------------------------------------------------
# Step 4: Show only high-value customers
# ------------------------------------------------------------

print("\nHIGH-VALUE CUSTOMERS")

high_value_customers.filter(
    F.col("customer_type") == "High Value"
).orderBy(
    F.desc("total_spending")
).show(truncate=False)





# ============================================================
# Exercise 6 - Anomaly Threshold Sensitivity
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 6 - ANOMALY THRESHOLD SENSITIVITY")
print("=" * 60)


# ------------------------------------------------------------
# Step 1: Test different z-score thresholds
# ------------------------------------------------------------

thresholds = [1.5, 2.0, 2.5, 3.0]

print("\nANOMALY COUNTS BY THRESHOLD")

for threshold in thresholds:

    anomaly_count = (
        anomaly_df
        .filter(F.abs(F.col("z")) > threshold)
        .count()
    )

    print(
        f"Threshold |z| > {threshold:.1f}: "
        f"{anomaly_count} anomalies"
    )




    # ------------------------------------------------------------
# Step 2: Show anomalies at each threshold
# ------------------------------------------------------------

for threshold in thresholds:

    print(
        f"\nTRANSACTIONS WITH |z| > {threshold:.1f}"
    )

    (
        anomaly_df
        .filter(F.abs(F.col("z")) > threshold)
        .select(
            "transaction_id",
            "customer",
            "category",
            "revenue",
            F.round("z", 3).alias("z")
        )
        .orderBy(F.desc(F.abs(F.col("z"))))
        .show(truncate=False)
    )




    # ============================================================
# Exercise 7 - RFM Marketing Actions
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 7 - RFM MARKETING ACTIONS")
print("=" * 60)


# ------------------------------------------------------------
# Step 1: Assign marketing action to each RFM segment
# ------------------------------------------------------------

marketing_df = (
    segmented_df
    .withColumn(
        "marketing_action",

        F.when(
            F.col("segment") == "Champions",
            "VIP rewards and exclusive offers"
        )

        .when(
            F.col("segment") == "Loyal",
            "Loyalty rewards and cross-sell offers"
        )

        .when(
            F.col("segment") == "Potential Loyalist",
            "Personalized offers to build loyalty"
        )

        .when(
            F.col("segment") == "New / Promising",
            "Welcome offers and product recommendations"
        )

        .when(
            F.col("segment") == "Hibernating / Lost",
            "Re-engagement campaign and special discount"
        )

        .otherwise(
            "General marketing campaign"
        )
    )
)


# ------------------------------------------------------------
# Step 2: Display recommendations
# ------------------------------------------------------------

print("\nCUSTOMER MARKETING RECOMMENDATIONS")

marketing_df.select(
    "customer",
    "recency_days",
    "frequency",
    "monetary",
    "segment",
    "marketing_action"
).orderBy(
    F.desc("rfm_sum")
).show(truncate=False)





# ============================================================
# Official Exercise 4 - High Quantity vs Payment Method
# ============================================================

print("\n" + "=" * 60)
print("OFFICIAL EXERCISE 4 - HIGH QUANTITY VS PAYMENT METHOD")
print("=" * 60)


# ------------------------------------------------------------
# Step 1: Count transactions by payment method and flag
# ------------------------------------------------------------

high_quantity_payment = (
    feature_df
    .groupBy("payment_method", "high_quantity")
    .count()
    .orderBy("payment_method", "high_quantity")
)

print("\nHIGH QUANTITY COUNTS BY PAYMENT METHOD")

high_quantity_payment.show(truncate=False)




# ------------------------------------------------------------
# Step 2: Crosstab / Pivot
# ------------------------------------------------------------

print("\nHIGH QUANTITY PAYMENT METHOD CROSSTAB")

high_quantity_pivot = (
    feature_df
    .groupBy("payment_method")
    .pivot("high_quantity", [0, 1])
    .count()
    .fillna(0)
    .withColumnRenamed("0", "normal_quantity")
    .withColumnRenamed("1", "high_quantity")
)

high_quantity_pivot.show(truncate=False)







# ============================================================
# Official Exercise 5 - RFM Threshold Tuning
# ============================================================

print("\n" + "=" * 60)
print("OFFICIAL EXERCISE 5 - RFM THRESHOLD TUNING")
print("=" * 60)


# ------------------------------------------------------------
# Step 1: Original segment distribution
# ------------------------------------------------------------

print("\nORIGINAL RFM SEGMENT COUNTS")

original_segment_counts = (
    segmented_df
    .groupBy("segment")
    .count()
    .orderBy(F.desc("count"))
)

original_segment_counts.show(truncate=False)



# ------------------------------------------------------------
# Step 2: Retune segmentation thresholds
# ------------------------------------------------------------

retuned_rfm_df = (
    segmented_df
    .withColumn(
        "retuned_segment",

        # Very strong customers
        F.when(
            (F.col("R") >= 3) &
            (F.col("F") >= 3) &
            (F.col("M") >= 3),
            "Champions"
        )

        # Recently active with good monetary value
        .when(
            (F.col("R") >= 3) &
            (F.col("M") >= 2),
            "Potential Loyalist"
        )

        # Good frequency even if not very recent
        .when(
            F.col("F") >= 3,
            "Loyal"
        )

        # Recent but lower engagement
        .when(
            F.col("R") >= 3,
            "New / Promising"
        )

        # Remaining customers
        .otherwise(
            "Hibernating / Lost"
        )
    )
)


print("\nRFM SEGMENT COMPARISON")

retuned_rfm_df.select(
    "customer",
    "R",
    "F",
    "M",
    "rfm_sum",
    "segment",
    "retuned_segment"
).orderBy(
    F.desc("rfm_sum")
).show(truncate=False)




print("\nRETUNED RFM SEGMENT COUNTS")

retuned_segment_counts = (
    retuned_rfm_df
    .groupBy("retuned_segment")
    .count()
    .orderBy(F.desc("count"))
)

retuned_segment_counts.show(truncate=False)



# ------------------------------------------------------------
# Step 5: Count customers whose segment changed
# ------------------------------------------------------------

changed_customers = (
    retuned_rfm_df
    .filter(
        F.col("segment") != F.col("retuned_segment")
    )
)

changed_count = changed_customers.count()

print(
    f"\nNumber of customers whose segment changed: "
    f"{changed_count}"
)

print("\nCUSTOMERS WITH CHANGED SEGMENTS")

changed_customers.select(
    "customer",
    "R",
    "F",
    "M",
    "segment",
    "retuned_segment"
).show(truncate=False)




# ============================================================
# Official Exercise 6 - Anomaly Threshold Comparison
# ============================================================

print("\n" + "=" * 60)
print("OFFICIAL EXERCISE 6 - ANOMALY THRESHOLD COMPARISON")
print("=" * 60)


# Original threshold: |z| > 2.0
original_anomalies = (
    anomaly_df
    .filter(F.abs(F.col("z")) > 2.0)
)

# New threshold: |z| > 1.5
new_anomalies = (
    anomaly_df
    .filter(F.abs(F.col("z")) > 1.5)
)


original_count = original_anomalies.count()
new_count = new_anomalies.count()


print(f"\nOriginal threshold |z| > 2.0: {original_count} anomalies")
print(f"New threshold      |z| > 1.5: {new_count} anomalies")


print("\nANOMALIES WITH NEW 1.5 THRESHOLD")

(
    new_anomalies
    .select(
        "transaction_id",
        "customer",
        "category",
        "revenue",
        F.round("z", 3).alias("z")
    )
    .orderBy(F.desc(F.abs(F.col("z"))))
    .show(truncate=False)
)


print("\nIMPACT OF THRESHOLD CHANGE")

if new_count > original_count:
    print(
        f"Lowering the threshold from 2.0 to 1.5 detected "
        f"{new_count - original_count} additional anomaly/anomalies."
    )
elif new_count == original_count:
    print(
        "Lowering the threshold from 2.0 to 1.5 did not "
        "identify any additional anomalies in this dataset."
    )
else:
    print("The new threshold detected fewer anomalies.")









# ============================================================
# Final Challenge - Region Health Score
# ============================================================

print("\n" + "=" * 60)
print("FINAL CHALLENGE - REGION HEALTH SCORE")
print("=" * 60)


# ------------------------------------------------------------
# Step 1: Calculate regional business metrics
# ------------------------------------------------------------

region_metrics = (
    feature_df
    .groupBy("region")
    .agg(
        F.round(F.sum("revenue"), 2).alias("total_revenue"),
        F.count("*").alias("transaction_count"),
        F.countDistinct("customer").alias("unique_customers"),
        F.round(F.avg("revenue"), 2).alias("avg_revenue")
    )
)

print("\nREGION BUSINESS METRICS")

region_metrics.orderBy(
    F.desc("total_revenue")
).show(truncate=False)


# ------------------------------------------------------------
# Step 2: Rank regions for each health indicator
# ------------------------------------------------------------

revenue_window = Window.orderBy(F.desc("total_revenue"))
transaction_window = Window.orderBy(F.desc("transaction_count"))
customer_window = Window.orderBy(F.desc("unique_customers"))


region_ranked = (
    region_metrics

    .withColumn(
        "revenue_rank",
        F.dense_rank().over(revenue_window)
    )

    .withColumn(
        "transaction_rank",
        F.dense_rank().over(transaction_window)
    )

    .withColumn(
        "customer_rank",
        F.dense_rank().over(customer_window)
    )
)


# ------------------------------------------------------------
# Step 3: Convert ranks into scores
#
# Rank 1 -> 4 points
# Rank 2 -> 3 points
# Rank 3 -> 2 points
# Rank 4 -> 1 point
# ------------------------------------------------------------

region_scored = (
    region_ranked

    .withColumn(
        "revenue_score",
        5 - F.col("revenue_rank")
    )

    .withColumn(
        "transaction_score",
        5 - F.col("transaction_rank")
    )

    .withColumn(
        "customer_score",
        5 - F.col("customer_rank")
    )

    .withColumn(
        "region_health_score",
        F.col("revenue_score")
        + F.col("transaction_score")
        + F.col("customer_score")
    )
)


# ------------------------------------------------------------
# Step 4: Classify regional health
# ------------------------------------------------------------

region_health = (
    region_scored
    .withColumn(
        "health_status",

        F.when(
            F.col("region_health_score") >= 10,
            "Strong"
        )

        .when(
            F.col("region_health_score") >= 7,
            "Moderate"
        )

        .otherwise("Needs Attention")
    )
)


# ------------------------------------------------------------
# Step 5: Display final result
# ------------------------------------------------------------

print("\nREGION HEALTH SCORE")

region_health.select(
    "region",
    "total_revenue",
    "transaction_count",
    "unique_customers",
    "revenue_score",
    "transaction_score",
    "customer_score",
    "region_health_score",
    "health_status"
).orderBy(
    F.desc("region_health_score")
).show(truncate=False)



