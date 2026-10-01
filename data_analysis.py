from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
    "type",
}


def validate_columns(df: pd.DataFrame, required_columns=None) -> None:
    """Validate that a DataFrame contains the required columns."""
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the wine-quality CSV file and validate its schema."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    validate_columns(df)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate observations and reset the index."""
    return df.drop_duplicates().reset_index(drop=True)


def filter_high_quality(
    df: pd.DataFrame,
    threshold: float = 7,
) -> pd.DataFrame:
    """Return wines whose quality score is at least the threshold."""
    if "quality" not in df.columns:
        raise ValueError("Missing required column: quality")

    return df[df["quality"] >= threshold].copy()


def summarize_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize average quality, alcohol content, and count by wine type."""
    required = {"type", "quality", "alcohol"}
    validate_columns(df, required)

    summary = (
        df.groupby("type")
        .agg(
            average_quality=("quality", "mean"),
            average_alcohol=("alcohol", "mean"),
            count=("quality", "count"),
        )
        .sort_index()
    )

    return summary


def high_quality_rate_by_type(
    df: pd.DataFrame,
    threshold: float = 7,
) -> pd.DataFrame:
    """Calculate the percentage of high-quality wines for each wine type."""
    required = {"type", "quality"}
    validate_columns(df, required)

    result = (
        df.assign(high_quality=df["quality"] >= threshold)
        .groupby("type")
        .agg(
            total_wines=("quality", "count"),
            high_quality_wines=("high_quality", "sum"),
            high_quality_rate=("high_quality", "mean"),
        )
        .sort_index()
    )

    result["high_quality_rate"] = result["high_quality_rate"] * 100

    return result


def detect_outliers_iqr(df: pd.DataFrame) -> pd.Series:
    """Count IQR-based outliers in each numeric column."""
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return pd.Series(dtype="int64", name="outlier_count")

    outlier_counts = {}

    for column in numeric_df.columns:
        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        is_outlier = (numeric_df[column] < lower_bound) | (
            numeric_df[column] > upper_bound
        )

        outlier_counts[column] = int(is_outlier.sum())

    return pd.Series(outlier_counts, name="outlier_count")


def create_visualization(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Create and save a scatter plot of alcohol content vs. wine quality."""
    required = {"alcohol", "quality"}
    validate_columns(df, required)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df["alcohol"],
        df["quality"],
        alpha=0.3,
    )

    ax.set_xlabel("Alcohol Content")
    ax.set_ylabel("Wine Quality")
    ax.set_title("Alcohol Content vs Wine Quality")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def create_high_quality_rate_visualization(
    rate_summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Create and save a bar chart of high-quality wine rates by type."""
    if "high_quality_rate" not in rate_summary.columns:
        raise ValueError("Missing required column: high_quality_rate")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    rate_summary["high_quality_rate"].plot(
        kind="bar",
        ax=ax,
    )

    ax.set_xlabel("Wine Type")
    ax.set_ylabel("High-Quality Wines (%)")
    ax.set_title("High-Quality Wine Rate by Type")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train and evaluate a Linear Regression model."""
    validate_columns(df)

    if len(df) < 5:
        raise ValueError("At least 5 observations are required for model training.")

    X = df.drop(columns=["quality", "type"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "model": model,
        "predictions": predictions,
        "y_test": y_test,
        "mae": mae,
        "r2": r2,
    }


def benchmark_pandas_polars(data_path: str | Path) -> dict:
    """Compare a small Pandas and Polars data-processing workflow."""
    data_path = Path(data_path)

    pandas_start = time.perf_counter()

    pandas_df = pd.read_csv(data_path)
    pandas_df = pandas_df.drop_duplicates()

    pandas_summary = summarize_by_type(pandas_df)

    pandas_time = time.perf_counter() - pandas_start

    polars_start = time.perf_counter()

    polars_df = pl.read_csv(data_path)
    polars_df = polars_df.unique()

    polars_summary = (
        polars_df.group_by("type")
        .agg(
            pl.col("quality").mean().alias("average_quality"),
            pl.col("alcohol").mean().alias("average_alcohol"),
            pl.len().alias("count"),
        )
        .sort("type")
    )

    polars_time = time.perf_counter() - polars_start

    return {
        "pandas_summary": pandas_summary,
        "polars_summary": polars_summary,
        "pandas_time": pandas_time,
        "polars_time": polars_time,
    }


def run_analysis(
    data_path: str | Path,
    plot_path: str | Path,
    quality_rate_plot_path: str | Path | None = None,
) -> dict:
    """Run the complete wine-quality analysis workflow."""
    original_df = load_data(data_path)
    cleaned_df = clean_data(original_df)

    high_quality = filter_high_quality(cleaned_df)
    type_summary = summarize_by_type(cleaned_df)
    high_quality_rate = high_quality_rate_by_type(cleaned_df)
    outlier_counts = detect_outliers_iqr(cleaned_df)

    create_visualization(cleaned_df, plot_path)

    if quality_rate_plot_path is not None:
        create_high_quality_rate_visualization(
            high_quality_rate,
            quality_rate_plot_path,
        )

    model_results = train_model(cleaned_df)

    return {
        "original_df": original_df,
        "cleaned_df": cleaned_df,
        "high_quality": high_quality,
        "type_summary": type_summary,
        "high_quality_rate": high_quality_rate,
        "outlier_counts": outlier_counts,
        "model_results": model_results,
        "plot_path": Path(plot_path),
        "quality_rate_plot_path": (
            Path(quality_rate_plot_path) if quality_rate_plot_path is not None else None
        ),
    }


def main() -> None:
    """Run the project analysis from the command line."""
    data_path = Path("data/wine_quality_merged.csv")
    plot_path = Path("wine_quality_scatter.png")
    quality_rate_plot_path = Path("high_quality_rate_by_type.png")

    results = run_analysis(
        data_path,
        plot_path,
        quality_rate_plot_path,
    )

    original_df = results["original_df"]
    cleaned_df = results["cleaned_df"]
    high_quality = results["high_quality"]
    type_summary = results["type_summary"]
    high_quality_rate = results["high_quality_rate"]
    outlier_counts = results["outlier_counts"]
    model_results = results["model_results"]

    print("First Five Rows:")
    print(original_df.head())

    print("\nShape:")
    print(original_df.shape)

    print("\nColumns:")
    print(original_df.columns)

    print("\nDataset Information:")
    original_df.info()

    print("\nSummary Statistics:")
    print(original_df.describe())

    print("\nMissing Values:")
    print(original_df.isnull().sum())

    print("\nNumber of Duplicate Rows:")
    print(original_df.duplicated().sum())

    print("\nShape After Removing Duplicates:")
    print(cleaned_df.shape)

    print("\nNumber of Duplicate Rows After Cleaning:")
    print(cleaned_df.duplicated().sum())

    print("\nHigh-Quality Wines (quality >= 7):")
    print(high_quality.head())

    print("\nNumber of High-Quality Wines:")
    print(len(high_quality))

    print("\nSummary by Wine Type:")
    print(type_summary)

    print("\nHigh-Quality Wine Rate by Type:")
    print(high_quality_rate)

    print("\nIQR-Based Outlier Counts:")
    print(outlier_counts)

    print(
        "\nOutlier Treatment: Outliers are reported but preserved "
        "because unusual measurements may represent valid wines."
    )

    print("\nMachine Learning Results:")
    print(f"Mean Absolute Error: {model_results['mae']:.3f}")
    print(f"R-squared: {model_results['r2']:.3f}")

    benchmark = benchmark_pandas_polars(data_path)

    print("\nPandas Result:")
    print(benchmark["pandas_summary"])

    print("\nPolars Result:")
    print(benchmark["polars_summary"])

    print("\nExecution Time:")
    print(f"Pandas: {benchmark['pandas_time']:.6f} seconds")
    print(f"Polars: {benchmark['polars_time']:.6f} seconds")


if __name__ == "__main__":
    main()
