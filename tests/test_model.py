import math

import pytest
from sklearn.linear_model import LinearRegression

from data_analysis import train_model


def test_train_model_returns_expected_results(sample_wine_df):
    """Model training should return the expected result objects and metrics."""
    results = train_model(sample_wine_df)

    assert isinstance(results["model"], LinearRegression)
    assert "predictions" in results
    assert "y_test" in results
    assert "mae" in results
    assert "r2" in results

    assert results["mae"] >= 0
    assert math.isfinite(results["mae"])
    assert math.isfinite(results["r2"])


def test_train_model_prediction_length_matches_test_set(sample_wine_df):
    """The number of predictions should match the number of test observations."""
    results = train_model(sample_wine_df)

    assert len(results["predictions"]) == len(results["y_test"])
    assert len(results["predictions"]) > 0


def test_train_model_is_reproducible(sample_wine_df):
    """Using the same random state should produce reproducible model results."""
    first_run = train_model(
        sample_wine_df,
        random_state=42,
    )

    second_run = train_model(
        sample_wine_df,
        random_state=42,
    )

    assert list(first_run["predictions"]) == pytest.approx(
        list(second_run["predictions"])
    )

    assert first_run["mae"] == pytest.approx(second_run["mae"])
    assert first_run["r2"] == pytest.approx(second_run["r2"])


def test_train_model_too_few_rows(sample_wine_df):
    """Model training should reject datasets that are too small."""
    small_df = sample_wine_df.iloc[:4].copy()

    with pytest.raises(
        ValueError,
        match="At least 5 observations",
    ):
        train_model(small_df)


def test_train_model_missing_required_column(sample_wine_df):
    """Model training should reject data with an incomplete schema."""
    invalid_df = sample_wine_df.drop(columns=["alcohol"])

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        train_model(invalid_df)
