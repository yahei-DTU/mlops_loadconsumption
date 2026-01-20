from torch.utils.data import Dataset
from mlops_loadconsumption.data import MyDataset
from tests import _PATH_DATA


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset(_PATH_DATA / 'raw_data')
    assert isinstance(dataset, Dataset)
