import pandas as pd
import pytest

from data_analysis import clean_data, filter_high_quality


def test_clean_data_removes_duplicates(sample_wine_df):
    """Duplicate rows should be removed."""
    duplicated_df = pd.concat(
        [sample_wine_df, sample_wine_df.iloc[[0]]],
        ignore_index=True,
    )

    cleaned_df = clean_data(duplicated_df)

    assert len(duplicated_df) == len(sample_wine_df) + 1
    assert len(cleaned_df) == len(sample_wine_df)
    assert cleaned_df.duplicated().sum() == 0


def test_clean_data_preserves_unique_rows(sample_wine_df):
    """Cleaning should not remove valid unique observations."""
    cleaned_df = clean_data(sample_wine_df)

    assert len(cleaned_df) == len(sample_wine_df)


def test_clean_data_resets_index(sample_wine_df):
    """The cleaned DataFrame should have a fresh sequential index."""
    modified_df = sample_wine_df.drop(index=[1])

    cleaned_df = clean_data(modified_df)

    assert list(cleaned_df.index) == list(range(len(cleaned_df)))


def test_filter_high_quality_default_threshold(sample_wine_df):
    """The default threshold should retain wines with quality >= 7."""
    filtered_df = filter_high_quality(sample_wine_df)

    assert len(filtered_df) == 2
    assert (filtered_df["quality"] >= 7).all()
    assert set(filtered_df["quality"]) == {7, 8}


def test_filter_high_quality_custom_threshold(sample_wine_df):
    """A custom quality threshold should be respected."""
    filtered_df = filter_high_quality(
        sample_wine_df,
        threshold=8,
    )

    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["quality"] == 8


def test_filter_high_quality_empty_result(sample_wine_df):
    """A threshold above all observed scores should return an empty DataFrame."""
    filtered_df = filter_high_quality(
        sample_wine_df,
        threshold=10,
    )

    assert filtered_df.empty
    assert list(filtered_df.columns) == list(sample_wine_df.columns)


def test_filter_high_quality_missing_quality_column(sample_wine_df):
    """Missing the quality column should raise a clear error."""
    invalid_df = sample_wine_df.drop(columns=["quality"])

    with pytest.raises(
        ValueError,
        match="Missing required column: quality",
    ):
        filter_high_quality(invalid_df)