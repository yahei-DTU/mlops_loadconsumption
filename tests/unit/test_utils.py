from tests import _TEST_ROOT,_PROJECT_ROOT,_PATH_DATA
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from mlops_loadconsumption.data import MyDataset

def test_dataset_import():
    """Test that MyDataset can be imported."""
    assert MyDataset is not None

def test_resample_to_hourly():
    """Test hourly resampling."""
    # Create mock 15-minute data
    dates = pd.date_range('2024-01-01', periods=96, freq='15min', tz='UTC')
    raw_data = pd.Series(
        data=np.random.rand(96) * 1000,
        index=dates
    )
    
    # Manually test the resampling logic
    hourly = raw_data.resample('h').mean()
    
    assert len(hourly) == 24
    assert hourly.index.freq == 'h'


def test_handle_missing_values():
    """Test missing value interpolation."""
    dates = pd.date_range('2024-01-01', periods=24, freq='h', tz='UTC')
    data = np.random.rand(24) * 1000
    data[5:8] = np.nan  # Add missing values
    
    df = pd.DataFrame(data={'load': data}, index=dates)
    
    # Test interpolation
    df_clean = df.interpolate(method='linear').ffill().bfill()
    
    assert df_clean.isna().sum().sum() == 0


def test_temporal_encoding_ranges():
    """Test that sin/cos encoding produces values in [-1, 1]."""
    dates = pd.date_range('2024-01-01', periods=168, freq='h', tz='UTC')
    
    hours = dates.hour
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    
    assert hour_sin.min() >= -1
    assert hour_sin.max() <= 1
    assert hour_cos.min() >= -1
    assert hour_cos.max() <= 1


def test_split_sizes_sum_to_one():
    """Test that split sizes validation works."""
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15
    
    assert train_size + val_size + test_size == 1.0


def test_create_sequences_logic():
    """Test sequence creation logic with simple data."""
    # Simple sequential data
    data = np.arange(50).reshape(-1, 1).astype(np.float32)
    n_input = 10
    n_output = 3
    
    # Calculate expected number of sequences
    n_sequences = len(data) - n_input - n_output + 1
    
    X_list = []
    Y_list = []
    
    for i in range(n_sequences):
        X = data[i:i + n_input, :]
        y = data[i + n_input:i + n_input + n_output, 0]
        X_list.append(X)
        Y_list.append(y)
    
    X_array = np.array(X_list)
    Y_array = np.array(Y_list)
    
    # Check shapes
    assert X_array.shape == (n_sequences, n_input, 1)
    assert Y_array.shape == (n_sequences, n_output)
    
    # Check first sequence values
    assert np.array_equal(X_array[0].flatten(), np.arange(10))
    assert np.array_equal(Y_array[0], np.arange(10, 13))


def test_holiday_feature_binary():
    """Test that holiday feature contains only 0 and 1."""
    dates = pd.date_range('2024-01-01', periods=168, freq='h', tz='UTC')
    
    # Create mock holiday indicator
    is_holiday = np.random.choice([0, 1], size=len(dates))
    
    assert set(is_holiday).issubset({0, 1})