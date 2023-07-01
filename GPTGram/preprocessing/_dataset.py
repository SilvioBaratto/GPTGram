import torch
import numpy as np
from torch.utils.data import Dataset
from ..config import Config as cfg

class GramDataset(Dataset):
    """
    A custom dataset class for loading data from a binary file.

    This class inherits from PyTorch's Dataset class and is designed to load data
    from a binary file using memory mapping for efficient access. The dataset is
    assumed to be a sequence of uint16 integers.

    Args:
        filepath (str): The file path of the binary data file.

    Examples:
        >>> dataset = WhatsDataset('path_to_your_file.bin')
        >>> sample_x, sample_y = dataset[0]

    """

    def __init__(self, filepath):
        """
        Initializes the dataset by loading the data from a file.

        Args:
            filepath (str): The file path of the binary data file.
        """
        # Load the data using memory mapping for efficient access
        self.data = np.memmap(filepath, dtype=np.uint16, mode='r')

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        The total number of samples is defined as the length of the data minus the block size.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data) - cfg.data.block_size

    def __getitem__(self, index):
        """
        Returns a sample from the dataset at the specified index.

        The sample consists of a pair of sequences x and y, where y is the same as x but shifted one step to the right.
        The sequences are cut from the data based on the current index and the block size defined in the configuration.

        Args:
            index (int): The index of the sample to return.

        Returns:
            tuple: A tuple containing two torch.LongTensor tensors, the input sequence (x) and the target sequence (y).

        """
        # Extract the input sequence x from the data
        x = torch.from_numpy((self.data[index:index+cfg.data.block_size]).astype(np.int64))
        # Extract the target sequence y from the data (shifted one step to the right)
        y = torch.from_numpy((self.data[index+1:index+1+cfg.data.block_size]).astype(np.int64))
        return x, y
