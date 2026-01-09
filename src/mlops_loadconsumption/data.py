from pathlib import Path
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

    def __init__(self,
                 data_path: Path,
                 start: pd.Timestamp = default_start_date,
                 end: pd.Timestamp = default_end_date,
                 api_key: str = default_api_key,
                 country: str = 'DK') -> None:
        self.data_path = data_path
        self.start = start
        self.end = end
        self.country = country
        self.client = EntsoePandasClient(api_key=api_key)
        
        self.api_request()

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def api_request(self) -> pd.DataFrame:
        """
        Pull load data from Denmark with specified timestamps.
        """
        print(f"Fetching load data for {self.country} from {self.start} to {self.end}...")
        
        # Query raw load data
        self.raw_data = self.client.query_load(self.country, start=self.start, end=self.end)
        
        # Resample to hourly resolution and convert to UTC
        self.hourly_load = self.raw_data.resample('h').mean()
        self.hourly_load.index = self.hourly_load.index.tz_convert('UTC')
        
        print(f"Successfully fetched {len(self.hourly_load)} hours of data")
        print(self.hourly_load.head())
        return self.hourly_load
        
        
    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
