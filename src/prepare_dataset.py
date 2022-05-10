"""Python file to make dataset structures."""

import os
from torch.utils.data import DataLoader

from data.dataset import reaction_record_dataset

RAW_DATASET_PATH = 'data/raw/'

def prep_dataset():
    """Prepare datasets."""

    train_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'train.txt')
    train_dataset = reaction_record_dataset(train_dataset_filepath, 'train')
    DataLoader(train_dataset)

    test_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'test.txt')
    test_dataset = reaction_record_dataset(test_dataset_filepath, 'test')
    DataLoader(test_dataset)

    val_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'valid.txt')
    val_dataset = reaction_record_dataset(val_dataset_filepath, 'val')
    DataLoader(val_dataset)


if __name__ == '__main__':
    prep_dataset()
