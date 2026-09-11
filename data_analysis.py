import time

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/wine_quality_merged.csv")

print(df.head())
print()

print("Shape:")
print(df.shape)
print()

print("Columns:")
print(df.columns)

# Inspect the Data

print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows

df = df.drop_duplicates().reset_index(drop=True)

print("\nShape After Removing Duplicates:")
print(df.shape)

print("\nNumber of Duplicate Rows After Cleaning:")
print(df.duplicated().sum())

# Basic Filtering

high_quality = df[df["quality"] >= 7]

print("\nHigh-Quality Wines (quality >= 7):")
print(high_quality.head())

print("\nNumber of High-Quality Wines:")
print(len(high_quality))

# Basic Grouping

type_summary = df.groupby("type").agg(
    average_quality=("quality", "mean"),
    average_alcohol=("alcohol", "mean"),
    count=("quality", "count")
)

print("\nSummary by Wine Type:")
print(type_summary)

# Visualization

plt.figure(figsize=(8, 5))

plt.scatter(
    df["alcohol"],
    df["quality"],
    alpha=0.3
)

plt.xlabel("Alcohol Content")
plt.ylabel("Wine Quality")
plt.title("Alcohol Content vs Wine Quality")

plt.tight_layout()
plt.savefig("wine_quality_scatter.png")
plt.show()

# Machine Learning

X = df.drop(columns=["quality", "type"])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMachine Learning Results:")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"R-squared: {r2:.3f}")

# Pandas vs Polars Performance Comparison

print("\nPandas vs Polars Performance Comparison:")

# Pandas
start = time.perf_counter()

pandas_df = pd.read_csv("data/wine_quality_merged.csv")
pandas_df = pandas_df.drop_duplicates()

pandas_summary = pandas_df.groupby("type").agg(
    average_quality=("quality", "mean"),
    average_alcohol=("alcohol", "mean"),
    count=("quality", "count")
)

pandas_time = time.perf_counter() - start


# Polars
start = time.perf_counter()

polars_df = pl.read_csv("data/wine_quality_merged.csv")
polars_df = polars_df.unique()

polars_summary = (
    polars_df
    .group_by("type")
    .agg(
        pl.col("quality").mean().alias("average_quality"),
        pl.col("alcohol").mean().alias("average_alcohol"),
        pl.len().alias("count")
    )
    .sort("type")
)

polars_time = time.perf_counter() - start


print("\nPandas Result:")
print(pandas_summary)

print("\nPolars Result:")
print(polars_summary)

print("\nExecution Time:")
print(f"Pandas: {pandas_time:.6f} seconds")
print(f"Polars: {polars_time:.6f} seconds")