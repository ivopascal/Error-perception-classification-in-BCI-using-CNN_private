import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataset import T_co

from settings import FEEDBACK_WINDOW_SIZE, CONTINUOUS_TEST_BATCH_SIZE
from src.util.util import milliseconds_to_samples


class DataModule(pl.LightningDataModule):
    def __init__(self, train_set, val_set, test_set, batch_size, test_batch_size=None):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size if test_batch_size else 1

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False, drop_last=True)


class ContinuousDataSet(IterableDataset):
    # This is currently not fit for training!

    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.generator())

    def generator(self):
        window_size = milliseconds_to_samples(FEEDBACK_WINDOW_SIZE)
        for x, y in zip(*self.data_source):
            for i in range(len(y[4]) - window_size):
                yield x[:, i: i + window_size], y[:4] + [y[4][i]]


class ContinuousDataModule(pl.LightningDataModule):
    def __init__(self, train_set, val_set, test_set, batch_size):
        super().__init__()
        self.train_set = ContinuousDataSet(train_set)
        self.val_set = ContinuousDataSet(val_set)
        self.test_set = ContinuousDataSet(test_set)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=CONTINUOUS_TEST_BATCH_SIZE, shuffle=False, drop_last=True)
