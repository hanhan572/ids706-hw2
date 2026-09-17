import pytest

from data_analysis import (
    benchmark_pandas_polars,
    create_visualization,
    summarize_by_type,
)


def test_summarize_by_type(sample_wine_df):
    """Grouping should calculate correct statistics for each wine type."""
    summary = summarize_by_type(sample_wine_df)

    assert set(summary.index) == {"red", "white"}

    assert summary.loc["red", "count"] == 3
    assert summary.loc["white", "count"] == 3

    assert summary.loc["red", "average_quality"] == pytest.approx(
        (5 + 6 + 6) / 3
    )

    assert summary.loc["white", "average_quality"] == pytest.approx(
        (8 + 5 + 7) / 3
    )

    assert summary.loc["red", "average_alcohol"] == pytest.approx(
        (9.5 + 10.0 + 10.8) / 3
    )

    assert summary.loc["white", "average_alcohol"] == pytest.approx(
        (11.5 + 9.0 + 12.0) / 3
    )


def test_summarize_by_type_missing_column(sample_wine_df):
    """Grouping should fail clearly if a required column is missing."""
    invalid_df = sample_wine_df.drop(columns=["alcohol"])

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        summarize_by_type(invalid_df)


def test_create_visualization_creates_file(tmp_path, sample_wine_df):
    """Visualization should create a non-empty image file."""
    output_path = tmp_path / "wine_plot.png"

    returned_path = create_visualization(
        sample_wine_df,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_create_visualization_missing_column(tmp_path, sample_wine_df):
    """Visualization should reject data without required columns."""
    invalid_df = sample_wine_df.drop(columns=["alcohol"])
    output_path = tmp_path / "invalid_plot.png"

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        create_visualization(
            invalid_df,
            output_path,
        )

def test_pandas_polars_produce_matching_results(tmp_path, sample_wine_df):
    """Pandas and Polars should produce equivalent grouped summaries."""
    data_path = tmp_path / "wine.csv"
    sample_wine_df.to_csv(data_path, index=False)

    results = benchmark_pandas_polars(data_path)

    pandas_summary = results["pandas_summary"]
    polars_summary = results["polars_summary"]

    polars_rows = {
        row["type"]: row
        for row in polars_summary.to_dicts()
    }

    assert set(pandas_summary.index) == set(polars_rows.keys())

    for wine_type in pandas_summary.index:
        assert pandas_summary.loc[
            wine_type, "average_quality"
        ] == pytest.approx(
            polars_rows[wine_type]["average_quality"]
        )

        assert pandas_summary.loc[
            wine_type, "average_alcohol"
        ] == pytest.approx(
            polars_rows[wine_type]["average_alcohol"]
        )

        assert pandas_summary.loc[
            wine_type, "count"
        ] == polars_rows[wine_type]["count"]

    assert results["pandas_time"] >= 0
    assert results["polars_time"] >= 0