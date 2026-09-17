import pandas as pd
import pytest

from data_analysis import load_data


def test_load_data_success(tmp_path, sample_wine_df):
    """A valid CSV should load successfully."""
    file_path = tmp_path / "wine.csv"
    sample_wine_df.to_csv(file_path, index=False)

    loaded_df = load_data(file_path)

    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == len(sample_wine_df)
    assert set(loaded_df.columns) == set(sample_wine_df.columns)


def test_load_data_file_not_found(tmp_path):
    """A missing CSV file should raise FileNotFoundError."""
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_data(missing_file)


def test_load_data_missing_required_column(tmp_path, sample_wine_df):
    """A dataset missing a required column should raise ValueError."""
    invalid_df = sample_wine_df.drop(columns=["quality"])

    file_path = tmp_path / "invalid.csv"
    invalid_df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_data(file_path)