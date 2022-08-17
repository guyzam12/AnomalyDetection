import math
import random

import pandas as pd
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch as th
from sklearn.preprocessing import StandardScaler

def load_data(
        data_file="",
        output_model_name="",
        batch_size=1,
        deterministic=False,
): #TODO: Replace description
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_file:
        raise ValueError("unspecified data directory")

    classes = None
    dataset = TableDataset(
        data_file,
        output_model_name,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader


class TableDataset(Dataset):
    def __init__(
            self,
            data_file,
            output_model_name,
    ):
        super().__init__()
        self.data_file = pd.read_csv(data_file,header=None)
        #self.data_file = self.data_file[self.data_file.iloc[:,-1] == 5]
        #self.data_file = self.data_file[self.data_file.columns[1:]]
        #self.data_file.to_pickle(output_model_name.replace('.pt','.pkl'))
        self.norm_data_file = self.normalize_data()
        print("hi")
        #self.normalize_data()
        #self.data_file = self.data_file.iloc[:,1:]
        '''
        del self.data_file['Id']
        del self.data_file['Species']
        '''

    def __len__(self):
        return th.tensor(len(self.norm_data_file))

    def __getitem__(self, idx):
        return th.tensor(self.norm_data_file.iloc[idx].values)

    def normalize_data(self):
        mean_per_column = self.data_file.mean()
        max_per_column = self.data_file.max()
        min_per_column = self.data_file.min()
        return (self.data_file - mean_per_column)/(max_per_column-min_per_column)