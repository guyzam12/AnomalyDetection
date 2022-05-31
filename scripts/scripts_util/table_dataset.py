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
    ):
        super().__init__()
        self.data_file = pd.read_csv(data_file)
        del self.data_file["Id"]
        self.label = self.data_file.pop('Species')
        self.label = pd.get_dummies(self.label).values.tolist()
        scaler = StandardScaler().fit(self.data_file)
        self.data_file = scaler.transform(self.data_file)
        # for column in self.data_file.columns:
        #     self.data_file[column] = (self.data_file[column]-self.data_file[column].min()) / (self.data_file[column].max()-self.data_file[column].min())



    def __len__(self):
        return th.tensor(len(self.data_file))

    def __getitem__(self, idx):
        # return th.tensor(self.data_file.iloc[idx]),th.tensor(self.label.iloc[idx])
        return th.tensor(self.data_file[idx]),th.tensor(self.label[idx])

    def get_label(self,idx):
        return th.tensor(self.label[idx])

