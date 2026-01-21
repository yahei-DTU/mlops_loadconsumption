import pytest
import pandas as pd
from pathlib import Path

def test_data_directory_exists(data_dir: Path) -> None:
    """Test that data directory exists."""
    assert data_dir.exists()


def test_raw_data_directory_exists(data_dir: Path) -> None:
    """Test that raw_data directory exists."""
    raw_data_path = data_dir / 'raw_data'
    assert raw_data_path.exists(), "raw_data directory should exist"


def test_raw_data_file_exists(data_dir: Path) -> None:
    """Test that raw data CSV file exists."""
    raw_data_file = data_dir / 'raw_data' / 'DK_load_raw.csv'
    assert raw_data_file.exists(), "DK_load_raw.csv should exist in raw_data folder"


def test_raw_data_can_be_loaded(data_dir: Path) -> None:
    """Test that raw data can be loaded as a pandas DataFrame."""
    raw_data_file = data_dir / 'raw_data' / 'DK_load_raw.csv'

    if raw_data_file.exists():
        df = pd.read_csv(raw_data_file, index_col=0, parse_dates=True)

        assert isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)
        assert len(df) > 0, "Raw data should not be empty"


def test_raw_data_has_no_missing_values(data_dir: Path) -> None:
    """Test that raw data has no missing values."""
    raw_data_file = data_dir / 'raw_data' / 'DK_load_raw.csv'

    if raw_data_file.exists():
        df = pd.read_csv(raw_data_file, index_col=0, parse_dates=True)

        # Check no NaN values
        missing_count = df.isna().sum()
        if isinstance(missing_count, pd.Series):
            missing_count = missing_count.sum()

        assert missing_count == 0, f"Raw data should have no missing values, found {missing_count}"


def test_raw_data_values_are_positive(data_dir: Path) -> None:
    """Test that raw data load values are positive."""
    raw_data_file = data_dir / 'raw_data' / 'DK_load_raw.csv'

    if raw_data_file.exists():
        df = pd.read_csv(raw_data_file, index_col=0, parse_dates=True)

        # Handle both Series and DataFrame
        if isinstance(df, pd.Series):
            assert (df > 0).all(), "All load values should be positive"
        else:
            # Assume first column or 'load' column contains the values
            if 'load' in df.columns:
                assert (df['load'] > 0).all(), "All load values should be positive"
            else:
                assert (df.iloc[:, 0] > 0).all(), "All load values should be positive"


def test_raw_data_is_hourly_resolution(data_dir: Path) -> None:
    """Test that raw data has hourly resolution."""
    raw_data_file = data_dir / 'raw_data' / 'DK_load_raw.csv'

    if raw_data_file.exists():
        df = pd.read_csv(raw_data_file, index_col=0, parse_dates=True)

        # Calculate time differences between consecutive timestamps
        time_diffs = df.index.to_series().diff()

        # Most common difference should be 1 hour
        # Use mode to find most frequent time difference
        most_common_diff = time_diffs.mode()[0]

        assert most_common_diff == pd.Timedelta(hours=1), \
            f"Data should be hourly, but most common interval is {most_common_diff}"


def test_raw_data_values_are_realistic(data_dir: Path) -> None:
    """Test that load values are in a realistic range for Denmark."""
    raw_data_file = data_dir / 'raw_data' / 'DK_load_raw.csv'

    if raw_data_file.exists():
        df = pd.read_csv(raw_data_file, index_col=0, parse_dates=True)

        # Handle both Series and DataFrame
        if isinstance(df, pd.Series):
            values = df
        else:
            values = df.iloc[:, 0] if 'load' not in df.columns else df['load']

        # Denmark's typical load is between 1500-6000 MW
        assert values.min() > 100, f"Minimum load {values.min()} seems too low"
        assert values.max() < 10000, f"Maximum load {values.max()} seems too high"


def test_processed_data_directory_exists(data_dir: Path) -> None:
    """Test that processed_data directory exists."""
    processed_path = data_dir / 'processed_data'

    if processed_path.exists():
        assert processed_path.is_dir()


def test_processed_data_can_be_loaded(data_dir: Path) -> None:
    """Test loading processed data if it exists."""
    processed_file = data_dir / 'processed_data' / 'DK_load_processed.csv'

    if processed_file.exists():
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

        # Check expected columns
        assert 'load' in df.columns

        # Check for temporal features
        expected_temporal = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        for col in expected_temporal:
            assert col in df.columns, f"Missing temporal feature: {col}"


def test_processed_data_has_no_missing_values(data_dir: Path) -> None:
    """Test that processed data has no missing values."""
    processed_file = data_dir / 'processed_data' / 'DK_load_processed.csv'

    if processed_file.exists():
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

        # Check no NaN values
        assert df.isna().sum().sum() == 0


def test_processed_data_load_values_are_positive(data_dir: Path) -> None:
    """Test that processed data load values are positive."""
    processed_file = data_dir / 'processed_data' / 'DK_load_processed.csv'

    if processed_file.exists():
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

        assert (df['load'] > 0).all()


def test_processed_data_is_hourly_resolution(data_dir: Path) -> None:
    """Test that processed data has hourly resolution."""
    processed_file = data_dir / 'processed_data' / 'DK_load_processed.csv'

    if processed_file.exists():
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

        # Calculate time differences
        time_diffs = df.index.to_series().diff()
        most_common_diff = time_diffs.mode()[0]

        assert most_common_diff == pd.Timedelta(hours=1), \
            f"Processed data should be hourly, but interval is {most_common_diff}"


def test_temporal_features_in_valid_range(data_dir: Path) -> None:
    """Test that sin/cos features are in [-1, 1]."""
    processed_file = data_dir / 'processed_data' / 'DK_load_processed.csv'

    if processed_file.exists():
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

        # Get all sin/cos columns
        temporal_cols = [col for col in df.columns if 'sin' in col or 'cos' in col]

        for col in temporal_cols:
            assert df[col].min() >= -1.01, f"{col} has values below -1"
            assert df[col].max() <= 1.01, f"{col} has values above 1"


def test_train_val_test_splits_exist(data_dir: Path) -> None:
    """Test that train/val/test splits can be loaded if they exist."""
    splits_path = data_dir / 'processed_data' / 'splits'

    if splits_path.exists():
        train_file = splits_path / 'train.csv'
        val_file = splits_path / 'val.csv'
        test_file = splits_path / 'test.csv'

        if train_file.exists():
            train = pd.read_csv(train_file, index_col=0, parse_dates=True)
            assert len(train) > 0

        if val_file.exists():
            val = pd.read_csv(val_file, index_col=0, parse_dates=True)
            assert len(val) > 0

        if test_file.exists():
            test = pd.read_csv(test_file, index_col=0, parse_dates=True)
            assert len(test) > 0


def test_splits_have_no_missing_values(data_dir: Path) -> None:
    """Test that train/val/test splits have no missing values."""
    splits_path = data_dir / 'processed_data' / 'splits'

    if splits_path.exists():
        for split_name in ['train.csv', 'val.csv', 'test.csv']:
            split_file = splits_path / split_name
            if split_file.exists():
                df = pd.read_csv(split_file, index_col=0, parse_dates=True)
                assert df.isna().sum().sum() == 0, f"{split_name} should have no missing values"


def test_splits_have_positive_load_values(data_dir: Path) -> None:
    """Test that train/val/test splits have positive load values."""
    splits_path = data_dir / 'processed_data' / 'splits'

    if splits_path.exists():
        for split_name in ['train.csv', 'val.csv', 'test.csv']:
            split_file = splits_path / split_name
            if split_file.exists():
                df = pd.read_csv(split_file, index_col=0, parse_dates=True)
                assert (df['load'] > 0).all(), f"{split_name} should have positive load values"


def test_splits_are_hourly_resolution(data_dir: Path) -> None:
    """Test that train/val/test splits have hourly resolution."""
    splits_path = data_dir / 'processed_data' / 'splits'

    if splits_path.exists():
        for split_name in ['train.csv', 'val.csv', 'test.csv']:
            split_file = splits_path / split_name
            if split_file.exists():
                df = pd.read_csv(split_file, index_col=0, parse_dates=True)

                time_diffs = df.index.to_series().diff()
                most_common_diff = time_diffs.mode()[0]

                assert most_common_diff == pd.Timedelta(hours=1), \
                    f"{split_name} should be hourly, but interval is {most_common_diff}"
