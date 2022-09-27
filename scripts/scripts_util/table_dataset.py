import math
import random

import pandas as pd
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset, BatchSampler
import torch as th
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import Sampler
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

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
        self.df = pd.read_csv(data_file, header=None)
        self.row_size = self.df.shape[1] - 1
        self.X = self.df.iloc[:,:-1]
        self.X = self.normalize_data(self.X)
        self.t = self.df.iloc[:,-1]
        self.create_norm_params_file(self.df,output_model_name)
        self.distances = self.knn(knn_small=1,knn_high=50)
        self.lof = self.lof(knn_small=10,knn_high=15)



    def __len__(self):
        return th.tensor(len(self.X))

    def __getitem__(self, idx):
        return th.tensor(self.X.iloc[idx].values),idx

    def create_norm_params_file(self,df,output_model_name):
        pkl = {'max_per_col': df.iloc[:,:-1].max(), 'min_per_col': df.iloc[:,:-1].min()}
        pkldf = pd.DataFrame(data=pkl)
        pkldf.to_pickle(output_model_name.replace('.pt', '.pkl'))


    def normalize_data(self, X):
        max_per_col, min_per_col = X.max(), X.min()
        X = 2*(X - min_per_col)/(max_per_col-min_per_col) - 1
        return X

    def knn(self, knn_small, knn_high):
        X = self.X
        nbrs = NearestNeighbors(n_neighbors=knn_high+1)
        nbrs.fit(X)
        distances, indexes = nbrs.kneighbors(X)
        distances_mean = []
        for i in range(knn_small+1,knn_high+1):
            cur_distances = np.mean(distances[:,1:i],axis=1)
            min_per_col, max_per_col = np.min(cur_distances), np.max(cur_distances)
            cur_distances = (cur_distances-min_per_col) / (max_per_col-min_per_col)
            distances_mean.append(cur_distances)
        return distances_mean

    def lof(self, knn_small, knn_high):
        lof_mean = []
        for i in range(knn_small, knn_high):
            clf = LocalOutlierFactor(n_neighbors=i)
            pred = clf.fit_predict(self.X)
            lof = -clf.negative_outlier_factor_
            lof = (lof - np.min(lof)) / (np.max(lof) - np.min(lof))
            lof_mean.append(lof)

        return lof_mean

    def get_labels(self,idx):
        return self.t[idx]

    def get_all_labels(self):
        return self.t

    def get_distances(self):
        return self.distances

    def get_row_size(self):
        return self.row_size

    def get_lof(self):
        return self.lof
