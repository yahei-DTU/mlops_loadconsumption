from pathlib import Path
import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient
import typer
from torch.utils.data import Dataset


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
        
        if data_path is None:
            data_path = self.default_data_path

        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True, parents=True)
        
        self.start = start
        self.end = end
        self.country = country
        self.client = EntsoePandasClient(api_key=api_key)

        self._fetch_api_data()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        pass
        
        
    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
    pass

    def _fetch_api_data(self):
        """
        Pull load data from Denmark with specified timestamps.
        """
        print(f"Fetching load data for {self.country} from {self.start} to {self.end}...")
        
        # Query raw load data
        try:
            self.raw_data = self.client.query_load(
                self.country, 
                start=self.start, 
                end=self.end
            )
            print(f"Successfully fetched {len(self.raw_data)} data points")
            
            # Save raw data
            raw_data_folder = self.data_path / 'raw_data'
            raw_data_folder.mkdir(parents=True, exist_ok=True)
            raw_data_file = raw_data_folder / f'{self.country}_load_raw.csv'
            self.raw_data.to_csv(raw_data_file)
            print(f"Raw data saved to {raw_data_file}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from ENTSO-E API: {str(e)}")
        
    def _resample_to_hourly(self):
        """Resample to hourly and convert to UTC."""
        hourly = self.raw_data.resample('h').mean()
        hourly.index = hourly.index.tz_convert('UTC')

        # Rename column explicitly
        hourly.columns = ['load']
        self.hourly_data = hourly

        print(f"Resampled to {len(self.hourly_data)} hours")

    def _handle_missing_values(self):
        """Interpolate missing values."""
        
        nan_count = self.hourly_data.isna().sum().sum()
        if nan_count > 0:
            print(f"Interpolating {nan_count} missing values")
            self.hourly_data = self.hourly_data.interpolate(method='linear').ffill().bfill()

    def _trigonometric_encoding(self):
        """
        Add trigonometric encoding of temporal features.
        
        Encodes hour, day of week, week, and month using sine/cosine 
        transformations to preserve cyclical nature.
        """
        if self.hourly_data is None:
            raise ValueError("No hourly data available. Process data first.")
        
        print("\nAdding temporal features with trigonometric encoding")
        
        # Create copy for feature engineering
        self.processed_data = self.hourly_data.copy()
        
        # Extract temporal components
        idx = self.processed_data.index
        hour = idx.hour
        day = idx.day
        month = idx.month
        dayofweek = idx.dayofweek
        week = idx.isocalendar().week.astype(int)
        
        # Hour encoding (0-23)
        self.processed_data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        self.processed_data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of month encoding (1-31)
        self.processed_data['day_sin'] = np.sin(2 * np.pi * day / 31)
        self.processed_data['day_cos'] = np.cos(2 * np.pi * day / 31)
        
        # Day of week encoding (0-6: Monday=0, Sunday=6)
        self.processed_data['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7)
        self.processed_data['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7)
        
        # Week of year encoding (1-53)
        self.processed_data['week_sin'] = np.sin(2 * np.pi * week / 53)
        self.processed_data['week_cos'] = np.cos(2 * np.pi * week / 53)
        
        # Month encoding (1-12)
        self.processed_data['month_sin'] = np.sin(2 * np.pi * month / 12)
        self.processed_data['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        temporal_features = [c for c in self.processed_data.columns if 'sin' in c or 'cos' in c]
        print(f"Added {len(temporal_features)} temporal features: {temporal_features}")
    

    def preprocess(self) -> None:
        """Preprocess the raw data and save it to the processed_data folder."""
        self._resample_to_hourly()
        self._handle_missing_values()
        self._trigonometric_encoding()
        
        # Save processed data
        processed_data_folder = self.data_path / 'processed_data'
        processed_data_folder.mkdir(parents=True, exist_ok=True)
        processed_data_file = processed_data_folder / f'{self.country}_load_processed.csv'
        self.processed_data.to_csv(processed_data_file)
        print(f"Processed data saved to {processed_data_file}")

def preprocess(data_path: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess()

if __name__ == "__main__":
    typer.run(preprocess)