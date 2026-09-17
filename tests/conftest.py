import pandas as pd
import pytest


@pytest.fixture
def sample_wine_df():
    """Create a small valid wine-quality DataFrame for testing."""
    return pd.DataFrame(
        {
            "fixed acidity": [7.0, 7.5, 8.0, 6.5, 7.2, 8.1],
            "volatile acidity": [0.30, 0.40, 0.25, 0.35, 0.28, 0.32],
            "citric acid": [0.30, 0.20, 0.40, 0.25, 0.35, 0.31],
            "residual sugar": [2.0, 2.5, 1.8, 3.0, 2.2, 2.4],
            "chlorides": [0.05, 0.06, 0.04, 0.05, 0.045, 0.055],
            "free sulfur dioxide": [20.0, 25.0, 18.0, 30.0, 22.0, 24.0],
            "total sulfur dioxide": [80.0, 90.0, 70.0, 100.0, 85.0, 88.0],
            "density": [0.995, 0.996, 0.994, 0.997, 0.9955, 0.9962],
            "pH": [3.20, 3.30, 3.10, 3.25, 3.15, 3.22],
            "sulphates": [0.50, 0.60, 0.55, 0.52, 0.58, 0.57],
            "alcohol": [9.5, 10.0, 11.5, 9.0, 12.0, 10.8],
            "quality": [5, 6, 8, 5, 7, 6],
            "type": ["red", "red", "white", "white", "white", "red"],
        }
    )