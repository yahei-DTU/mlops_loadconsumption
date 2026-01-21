from pathlib import Path
import logging
import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient
from torch.utils.data import Dataset
import holidays
from sklearn.model_selection import train_test_split
import tensorflow as tf
import hydra
from omegaconf import DictConfig

# Configure logger
logger = logging.getLogger(__name__)

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self,
                 n_input_timesteps: int,
                 n_output_timesteps: int,
                 train_size: float,
                 val_size: float,
                 test_size: float,
                 data_path: Path = None,
                 start: pd.Timestamp = None,
                 end: pd.Timestamp = None,
                 api_key: str = None,
                 country: str = None) -> None:
        """
        Initialize MyDataset with config parameters.

        Args:
            n_input_timesteps: Number of input timesteps for sequences
            n_output_timesteps: Number of output timesteps for sequences
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            test_size: Proportion of data for testing
            data_path: Path to data folder (default: <root>/data)
            start: Start date for data (default: 2023-01-01)
            end: End date for data (default: 2025-01-01)
            api_key: ENTSO-E API key (default: from config)
            country: Country code (default: 'DK')
        """
        # Build constants in constructor
        self.n_input_timesteps = n_input_timesteps
        self.n_output_timesteps = n_output_timesteps
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # Set defaults if not provided
        self.root = Path(__file__).resolve().parents[2]
        self.data_path = Path(data_path) if data_path is not None else self.root / "data"
        self.data_path.mkdir(exist_ok=True, parents=True)

        self.start = start if start is not None else pd.Timestamp('20230101', tz='UTC')
        self.end = end if end is not None else pd.Timestamp('20250101', tz='UTC')
        self.api_key = api_key if api_key is not None else '1ec78127-e12b-4cb2-a9fb-1258e4d5622a'
        self.country = country if country is not None else 'DK'

        logger.info(f"Initializing MyDataset for {self.country}")
        logger.info(f"Input timesteps: {self.n_input_timesteps}, Output timesteps: {self.n_output_timesteps}")
        logger.info(f"Split sizes - Train: {self.train_size}, Val: {self.val_size}, Test: {self.test_size}")

        # Initialize API client
        self.client = EntsoePandasClient(api_key=self.api_key)

        # Build holidays set
        country_holidays = set(holidays.Denmark(years=range(self.start.year, self.end.year + 1)).keys())

        # Add Sundays as holidays
        date_range = pd.date_range(start=self.start.date(), end=self.end.date(), freq='D')
        sundays = [d.date() for d in date_range if d.dayofweek == 6]  # 6 = Sunday
        country_holidays.update(sundays)
        self.holidays = country_holidays

        self._fetch_api_data()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        pass

    def __getitem__(self, index: int) -> pd.Series:
        """Return a given sample from the dataset."""
        pass

    def _fetch_api_data(self) -> None:
        """Pull load data from Denmark with specified timestamps."""
        logger.info(f"Fetching load data for {self.country} from {self.start} to {self.end}...")

        try:
            self.raw_data = self.client.query_load(
                self.country,
                start=self.start,
                end=self.end
            )
            logger.info(f"Successfully fetched {len(self.raw_data)} data points")

            raw_data_folder = self.data_path / 'raw_data'
            raw_data_folder.mkdir(parents=True, exist_ok=True)
            raw_data_file = raw_data_folder / f'{self.country}_load_raw.csv'
            self.raw_data.to_csv(raw_data_file)
            logger.info(f"Raw data saved to {raw_data_file}")

        except Exception as e:
            logger.error(f"Failed to fetch data from ENTSO-E API: {str(e)}")
            raise RuntimeError(f"Failed to fetch data from ENTSO-E API: {str(e)}")

    def _resample_to_hourly(self) -> None:
        """Resample to hourly and convert to UTC."""
        logger.info("Resampling data to hourly frequency")
        hourly = self.raw_data.resample('h').mean()
        hourly.index = hourly.index.tz_convert('UTC')
        hourly.columns = ['load']
        self.hourly_data = hourly
        logger.info(f"Resampled to {len(self.hourly_data)} hours")

    def _handle_missing_values(self) -> None:
        """Interpolate missing values."""
        logger.info("Handling missing values")

        nan_count = self.hourly_data.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} missing values, interpolating...")
            self.hourly_data = self.hourly_data.interpolate(method='linear').ffill().bfill()
            logger.info("Missing values interpolated successfully")
        else:
            logger.debug("No missing values found")

    def _trigonometric_encoding(self) -> None:
        """Add trigonometric encoding of temporal features."""
        logger.info("Adding temporal features with trigonometric encoding")

        if self.hourly_data is None:
            logger.error("No hourly data available")
            raise ValueError("No hourly data available. Process data first.")

        self.processed_data = self.hourly_data.copy()

        idx = self.processed_data.index
        hour = idx.hour
        day = idx.day
        month = idx.month
        dayofweek = idx.dayofweek
        week = idx.isocalendar().week.astype(int)

        self.processed_data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        self.processed_data['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        self.processed_data['day_sin'] = np.sin(2 * np.pi * day / 31)
        self.processed_data['day_cos'] = np.cos(2 * np.pi * day / 31)

        self.processed_data['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7)
        self.processed_data['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7)

        self.processed_data['week_sin'] = np.sin(2 * np.pi * week / 53)
        self.processed_data['week_cos'] = np.cos(2 * np.pi * week / 53)

        self.processed_data['month_sin'] = np.sin(2 * np.pi * month / 12)
        self.processed_data['month_cos'] = np.cos(2 * np.pi * month / 12)

        temporal_features = [c for c in self.processed_data.columns if 'sin' in c or 'cos' in c]
        logger.info(f"Added {len(temporal_features)} temporal features: {temporal_features}")

    def _add_holiday_feature(self) -> None:
        """Add holiday indicator feature."""
        logger.info("Adding holiday feature")

        if self.processed_data is None:
            logger.error("No processed data available")
            raise ValueError("No processed data available. Run trigonometric encoding first.")

        idx = self.processed_data.index
        holidays_timestamps = pd.to_datetime(list(self.holidays))
        self.processed_data['is_holiday'] = pd.to_datetime(idx.date).isin(holidays_timestamps).astype(int)
        logger.info("Holiday feature added successfully")

    def _split_data(self) -> None:
        """
        Split processed data into train, validation, and test sets using instance attributes.
        """

        if self.processed_data is None:
            logger.error("No processed data available")
            raise ValueError("No processed data available. Run preprocessing first.")

        if not (self.train_size + self.val_size + self.test_size == 1.0):
            logger.error("Train, validation, and test sizes must sum to 1.0")
            raise ValueError("Train, validation, and test sizes must sum to 1.0")

        # First split: train + val vs test
        self.train_val_data, self.test_data = train_test_split(
            self.processed_data,
            test_size=self.test_size,
            shuffle=False
        )

        # Second split: train vs val
        val_ratio = self.val_size / (self.train_size + self.val_size)
        self.train_data, self.val_data = train_test_split(
            self.train_val_data,
            test_size=val_ratio,
            shuffle=False
        )

        # Log data sizes
        logger.info(f"Train set dates: {self.train_data.index[0]} to {self.train_data.index[-1]}")
        logger.info(f"Validation set dates: {self.val_data.index[0]} to {self.val_data.index[-1]}")
        logger.info(f"Test set dates: {self.test_data.index[0]} to {self.test_data.index[-1]}")

    def _save_splits(self) -> None:
        """Save train, validation, and test sets to CSV files."""
        splits_folder = self.data_path / 'processed_data' / 'splits'
        splits_folder.mkdir(parents=True, exist_ok=True)
        self.train_data.to_csv(splits_folder / 'train.csv')
        self.val_data.to_csv(splits_folder / 'val.csv')
        self.test_data.to_csv(splits_folder / 'test.csv')

    def _create_sequences(self, data: np.ndarray) -> tuple:
        """
        Convert time series data into supervised learning format (X, y sequences).

        Args:
            data: Input array of shape (n_samples, n_features)

        Returns:
            X_tensor: Input sequences tensor
            Y_tensor: Output sequences tensor
        """
        logger.info(f"Creating sequences with n_input={self.n_input_timesteps}, n_output={self.n_output_timesteps}")

        X_list, Y_list = [], []

        for i in range(len(data) - self.n_input_timesteps - self.n_output_timesteps + 1):
            X = data[i:i + self.n_input_timesteps, :]  # Keep all features (multivariate)
            y = data[i + self.n_input_timesteps:i + self.n_input_timesteps + self.n_output_timesteps, 0]  # Only load (first column)

            X_list.append(X)
            Y_list.append(y)

        X_array = np.array(X_list)
        Y_array = np.array(Y_list)

        X_tensor = tf.convert_to_tensor(X_array, dtype=tf.float32)
        Y_tensor = tf.convert_to_tensor(Y_array, dtype=tf.float32)

        logger.info(f"Created {len(X_list)} sequences - X shape: {X_tensor.shape}, Y shape: {Y_tensor.shape}")

        return X_tensor, Y_tensor

    def get_train_val_test_sequences(self) -> dict:
        """
        Create sequences for train, validation, and test sets.

        Returns:
            Dictionary with train, val, test X and y tensors
        """
        logger.info("Creating sequences for all data splits")

        X_train, y_train = self._create_sequences(np.array(self.train_data))
        X_val, y_val = self._create_sequences(np.array(self.val_data))
        X_test, y_test = self._create_sequences(np.array(self.test_data))

        sequences = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }

        logger.info("Sequences created successfully")
        return sequences

    def preprocess(self) -> None:
        """Preprocess the raw data and save it to the processed_data folder."""
        logger.info("Starting preprocessing pipeline")

        self._resample_to_hourly()
        self._handle_missing_values()
        self._trigonometric_encoding()
        self._add_holiday_feature()

        processed_data_folder = self.data_path / 'processed_data'
        processed_data_folder.mkdir(parents=True, exist_ok=True)
        processed_data_file = processed_data_folder / f'{self.country}_load_processed.csv'
        self.processed_data.to_csv(processed_data_file)
        logger.info(f"Processed data saved to {processed_data_file}")

        # Split and save data
        self._split_data()
        self._save_splits()

        logger.info("Preprocessing pipeline completed successfully")

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def preprocess(cfg: DictConfig) -> None:
    """Preprocess data using config parameters."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Preprocessing data...")

    # Create dataset with config parameters
    dataset = MyDataset(
        n_input_timesteps=cfg.data.n_input_timesteps,
        n_output_timesteps=cfg.data.n_output_timesteps,
        train_size=cfg.split.train_size,
        val_size=cfg.split.val_size,
        test_size=cfg.split.test_size,
        api_key=cfg.api.key,
        country=cfg.api.country
    )
    dataset.preprocess()

if __name__ == "__main__":
    preprocess()
