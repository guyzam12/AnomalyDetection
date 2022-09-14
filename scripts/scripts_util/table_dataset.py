import math
import random

import pandas as pd
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset, BatchSampler
import torch as th
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import Sampler
from sklearn.neighbors import NearestNeighbors

def load_data(
        data_obj,
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
    if not data_obj:
        raise ValueError("unspecified data object")

    if deterministic:
        loader = DataLoader(
            data_obj, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            data_obj, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
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
        self.data_file = pd.read_csv(data_file, header=None)
        df = self.data_file
        #df = self.data_file.iloc[:,1:]
        self.data_file_labels = df.iloc[:,-1]
        df = df.iloc[:,:-1]
        self.data_file = df
        self.max_per_col = self.data_file.max()
        self.min_per_col = self.data_file.min()
        self.mean_per_col = self.data_file.mean()
        self.std_per_col = self.data_file.std(ddof=0)
        pkl = {'max_per_col': self.max_per_col, 'min_per_col': self.min_per_col}
        pkldf = pd.DataFrame(data=pkl)
        pkldf.to_pickle(output_model_name.replace('.pt', '.pkl'))
        self.norm_data_file = self.normalize_data()
        self.distances = self.knn()

    def __len__(self):
        return th.tensor(len(self.norm_data_file))

    def __getitem__(self, idx):
        return th.tensor(self.norm_data_file.iloc[idx].values),idx

    def normalize_data(self):
        #norm_data_file = (self.data_file - self.mean_per_col)/self.std_per_col
        norm_data_file = 2*(self.data_file - self.min_per_col)/(self.max_per_col-self.min_per_col) - 1
        return norm_data_file
        #return (self.data_file - min_per_column)/(max_per_column-min_per_column)

    def knn(self):
        df = self.norm_data_file
        #nbrs = NearestNeighbors(n_neighbors=50)
        #nbrs.fit(df)
        #distances, indexes = nbrs.kneighbors(df)
        #distances = pd.DataFrame(distances)
        #distance_mean = th.zeros(distances.shape[0])
        distances_mean = 0
        count=0
        for i in range(10,51):
            nbrs = NearestNeighbors(n_neighbors=i)
            nbrs.fit(df)
            distances, indexes = nbrs.kneighbors(df)
            distances_org = distances
            distances = distances*np.exp(-distances/2)
            distances = np.mean(distances,axis=1)
            dist = distances
            #dist = (dist - min(dist)) / (max(dist) - min(dist))
            distances_mean += dist
            count += 1
        distances_mean /= count
        #distances_mean = th.tensor(distances.iloc[:,1:].mean(axis=1))
        distances_mean = (distances_mean - min(distances_mean))/(max(distances_mean)-min(distances_mean))
        return th.tensor(distances_mean)

    def get_labels(self,idx):
        return self.data_file_labels[idx]

    def get_all_labels(self):
        return self.data_file_labels

    def get_distances(self):
        return self.distances