from pathlib import Path

import pandas as pd
import pytest

from data_analysis import run_analysis


def test_complete_analysis_workflow(tmp_path, sample_wine_df):
    """
    The complete workflow should load, clean, analyze, visualize,
    train, predict, and evaluate successfully.
    """
    # Add one duplicate row so the integration test also verifies cleaning.
    input_df = pd.concat(
        [sample_wine_df, sample_wine_df.iloc[[0]]],
        ignore_index=True,
    )

    data_path = tmp_path / "test_wine.csv"
    plot_path = tmp_path / "test_plot.png"

    input_df.to_csv(data_path, index=False)

    results = run_analysis(
        data_path=data_path,
        plot_path=plot_path,
    )

    # Data loading
    assert len(results["original_df"]) == 7

    # Data cleaning
    assert len(results["cleaned_df"]) == 6
    assert results["cleaned_df"].duplicated().sum() == 0

    # Filtering
    assert len(results["high_quality"]) == 2
    assert (results["high_quality"]["quality"] >= 7).all()

    # Grouping
    assert set(results["type_summary"].index) == {"red", "white"}

    # Visualization
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0

    # Machine learning
    model_results = results["model_results"]

    assert len(model_results["predictions"]) == len(model_results["y_test"])
    assert model_results["mae"] >= 0


def test_real_project_dataset_workflow(tmp_path):
    """
    The complete workflow should reproduce the expected results
    using the actual project dataset.
    """
    project_root = Path(__file__).resolve().parents[1]

    data_path = project_root / "data" / "wine_quality_merged.csv"
    plot_path = tmp_path / "real_dataset_plot.png"

    results = run_analysis(
        data_path=data_path,
        plot_path=plot_path,
    )

    # Original data
    assert results["original_df"].shape == (6497, 13)

    # Cleaning reproducibility
    assert results["cleaned_df"].shape == (5320, 13)
    assert results["cleaned_df"].duplicated().sum() == 0

    # Filtering reproducibility
    assert len(results["high_quality"]) == 1009

    # Grouping reproducibility
    summary = results["type_summary"]

    assert summary.loc["red", "count"] == 1359
    assert summary.loc["white", "count"] == 3961

    assert summary.loc["red", "average_quality"] == pytest.approx(
        5.623252,
        abs=0.000001,
    )

    assert summary.loc["white", "average_quality"] == pytest.approx(
        5.854835,
        abs=0.000001,
    )

    # Model reproducibility
    model_results = results["model_results"]

    assert model_results["mae"] == pytest.approx(
        0.561,
        abs=0.001,
    )

    assert model_results["r2"] == pytest.approx(
        0.302,
        abs=0.001,
    )

    # Visualization
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0