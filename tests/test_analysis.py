import pandas as pd
import pytest

from data_analysis import (
    benchmark_pandas_polars,
    create_high_quality_rate_visualization,
    create_visualization,
    detect_outliers_iqr,
    high_quality_rate_by_type,
    summarize_by_type,
)


def test_summarize_by_type(sample_wine_df):
    summary = summarize_by_type(sample_wine_df)

    assert set(summary.index) == {"red", "white"}

    assert summary.loc["red", "count"] == 3
    assert summary.loc["white", "count"] == 3

    assert summary.loc["red", "average_quality"] == pytest.approx((5 + 6 + 6) / 3)
    assert summary.loc["white", "average_quality"] == pytest.approx((8 + 5 + 7) / 3)

    assert summary.loc["red", "average_alcohol"] == pytest.approx(
        sample_wine_df.loc[
            sample_wine_df["type"] == "red",
            "alcohol",
        ].mean()
    )

    assert summary.loc["white", "average_alcohol"] == pytest.approx(
        sample_wine_df.loc[
            sample_wine_df["type"] == "white",
            "alcohol",
        ].mean()
    )


def test_summarize_by_type_missing_column(sample_wine_df):
    invalid_df = sample_wine_df.drop(columns=["alcohol"])

    with pytest.raises(ValueError, match="Missing required columns"):
        summarize_by_type(invalid_df)


def test_high_quality_rate_by_type(sample_wine_df):
    result = high_quality_rate_by_type(sample_wine_df)

    assert result.loc["red", "total_wines"] == 3
    assert result.loc["white", "total_wines"] == 3

    assert result.loc["red", "high_quality_wines"] == 0
    assert result.loc["white", "high_quality_wines"] == 2

    assert result.loc["red", "high_quality_rate"] == pytest.approx(0.0)
    assert result.loc["white", "high_quality_rate"] == pytest.approx(2 / 3 * 100)


def test_high_quality_rate_custom_threshold(sample_wine_df):
    result = high_quality_rate_by_type(
        sample_wine_df,
        threshold=8,
    )

    assert result.loc["red", "high_quality_wines"] == 0
    assert result.loc["white", "high_quality_wines"] == 1


def test_high_quality_rate_missing_column(sample_wine_df):
    invalid_df = sample_wine_df.drop(columns=["type"])

    with pytest.raises(ValueError, match="Missing required columns"):
        high_quality_rate_by_type(invalid_df)


def test_detect_outliers_iqr():
    df = pd.DataFrame(
        {
            "measurement": [1, 2, 2, 2, 3, 100],
            "type": ["red", "red", "white", "white", "red", "white"],
        }
    )

    result = detect_outliers_iqr(df)

    assert result["measurement"] == 1
    assert result.name == "outlier_count"


def test_detect_outliers_iqr_no_numeric_columns():
    df = pd.DataFrame(
        {
            "type": ["red", "white"],
        }
    )

    result = detect_outliers_iqr(df)

    assert result.empty
    assert result.name == "outlier_count"


def test_create_visualization_creates_file(sample_wine_df, tmp_path):
    output_path = tmp_path / "wine_plot.png"

    result_path = create_visualization(
        sample_wine_df,
        output_path,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_create_visualization_missing_column(sample_wine_df, tmp_path):
    invalid_df = sample_wine_df.drop(columns=["alcohol"])
    output_path = tmp_path / "wine_plot.png"

    with pytest.raises(ValueError, match="Missing required columns"):
        create_visualization(
            invalid_df,
            output_path,
        )


def test_create_high_quality_rate_visualization(
    sample_wine_df,
    tmp_path,
):
    rate_summary = high_quality_rate_by_type(sample_wine_df)
    output_path = tmp_path / "high_quality_rate.png"

    result_path = create_high_quality_rate_visualization(
        rate_summary,
        output_path,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_create_high_quality_rate_visualization_missing_column(tmp_path):
    invalid_summary = pd.DataFrame(
        {
            "total_wines": [3, 3],
        },
        index=["red", "white"],
    )

    output_path = tmp_path / "high_quality_rate.png"

    with pytest.raises(
        ValueError,
        match="Missing required column: high_quality_rate",
    ):
        create_high_quality_rate_visualization(
            invalid_summary,
            output_path,
        )


def test_pandas_polars_produce_matching_results(sample_wine_df, tmp_path):
    data_path = tmp_path / "wine.csv"
    sample_wine_df.to_csv(data_path, index=False)

    results = benchmark_pandas_polars(data_path)

    pandas_summary = results["pandas_summary"].reset_index()

    polars_summary = pd.DataFrame(results["polars_summary"].to_dicts())

    pandas_summary = pandas_summary.sort_values("type").reset_index(drop=True)
    polars_summary = polars_summary.sort_values("type").reset_index(drop=True)

    assert list(pandas_summary["type"]) == list(polars_summary["type"])

    assert list(pandas_summary["count"]) == list(polars_summary["count"])

    assert list(pandas_summary["average_quality"]) == pytest.approx(
        list(polars_summary["average_quality"])
    )

    assert list(pandas_summary["average_alcohol"]) == pytest.approx(
        list(polars_summary["average_alcohol"])
    )

    assert results["pandas_time"] >= 0
    assert results["polars_time"] >= 0
