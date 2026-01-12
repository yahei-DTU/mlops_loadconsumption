from pathlib import Path
import logging
import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient
import typer
from torch.utils.data import Dataset
import holidays
from sklearn.model_selection import train_test_split

# Configure logger
logger = logging.getLogger(__name__)



class MyDataset(Dataset):
    """My custom dataset."""
    # Class-level constants
    default_api_key = '1ec78127-e12b-4cb2-a9fb-1258e4d5622a'
    default_start_date = pd.Timestamp('20230101', tz='UTC')
    default_end_date = pd.Timestamp('20250101', tz='UTC')
    root = Path(__file__).resolve().parents[2]
    default_data_path = root / "data"

    def __init__(self,
                 data_path: Path,
                 start: pd.Timestamp = default_start_date,
                 end: pd.Timestamp = default_end_date,
                 api_key: str = default_api_key,
                 country: str = 'DK') -> None:

        logger.info(f"Initializing MyDataset for {country}")

        if data_path is None:
            data_path = self.default_data_path

        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True, parents=True)

        self.start = start
        self.end = end
        self.country = country
        self.client = EntsoePandasClient(api_key=api_key)
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


    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
    pass

    def _fetch_api_data(self):
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

    def _resample_to_hourly(self):
        """Resample to hourly and convert to UTC."""
        logger.info("Resampling data to hourly frequency")
        hourly = self.raw_data.resample('h').mean()
        hourly.index = hourly.index.tz_convert('UTC')
        hourly.columns = ['load']
        self.hourly_data = hourly
        logger.info(f"Resampled to {len(self.hourly_data)} hours")

    def _handle_missing_values(self):
        """Interpolate missing values."""
        logger.info("Handling missing values")

        nan_count = self.hourly_data.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} missing values, interpolating...")
            self.hourly_data = self.hourly_data.interpolate(method='linear').ffill().bfill()
            logger.info("Missing values interpolated successfully")
        else:
            logger.debug("No missing values found")

    def _trigonometric_encoding(self):
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

        logger.debug("Encoding hour features")
        self.processed_data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        self.processed_data['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        logger.debug("Encoding day features")
        self.processed_data['day_sin'] = np.sin(2 * np.pi * day / 31)
        self.processed_data['day_cos'] = np.cos(2 * np.pi * day / 31)

        logger.debug("Encoding day of week features")
        self.processed_data['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7)
        self.processed_data['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7)

        logger.debug("Encoding week features")
        self.processed_data['week_sin'] = np.sin(2 * np.pi * week / 53)
        self.processed_data['week_cos'] = np.cos(2 * np.pi * week / 53)

        logger.debug("Encoding month features")
        self.processed_data['month_sin'] = np.sin(2 * np.pi * month / 12)
        self.processed_data['month_cos'] = np.cos(2 * np.pi * month / 12)

        temporal_features = [c for c in self.processed_data.columns if 'sin' in c or 'cos' in c]
        logger.info(f"Added {len(temporal_features)} temporal features: {temporal_features}")

    def _add_holiday_feature(self):
        """Add holiday indicator feature."""
        logger.info("Adding holiday feature")

        if self.processed_data is None:
            logger.error("No processed data available")
            raise ValueError("No processed data available. Run trigonometric encoding first.")

        idx = self.processed_data.index
        holidays_timestamps = pd.to_datetime(list(self.holidays))
        self.processed_data['is_holiday'] = pd.to_datetime(idx.date).isin(holidays_timestamps).astype(int)
        logger.info("Holiday feature added successfully")

    def _split_data(self, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15) -> None:
        """
        Split processed data into train, validation, and test sets.

        Args:
            train_size: Proportion of data for training (default: 0.7)
            val_size: Proportion of data for validation (default: 0.15)
            test_size: Proportion of data for testing (default: 0.15)
        """

        if self.processed_data is None:
            logger.error("No processed data available")
            raise ValueError("No processed data available. Run preprocessing first.")

        if not (train_size + val_size + test_size == 1.0):
            logger.error("Train, validation, and test sizes must sum to 1.0")
            raise ValueError("Train, validation, and test sizes must sum to 1.0")

        # First split: train + val vs test
        self.train_val_data, self.test_data = train_test_split(
            self.processed_data,
            test_size=test_size,
            shuffle=False
        )

        # Second split: train vs val
        val_ratio = val_size / (train_size + val_size)
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
        self.train_data.to_csv(splits_folder / 'train.csv')
        self.val_data.to_csv(splits_folder / 'val.csv')
        self.test_data.to_csv(splits_folder / 'test.csv')

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

def preprocess(data_path: Path) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess()

if __name__ == "__main__":
    typer.run(preprocess)
