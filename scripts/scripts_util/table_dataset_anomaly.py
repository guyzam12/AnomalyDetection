import math
import random

import pandas as pd
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset, BatchSampler
import torch as th
import re
import glob
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import Sampler
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest

PROJECT_PATH = "/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion"

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

class TableDatasetAnomaly(Dataset):
    def __init__(
            self,
            data_file,
            train_data_file,
    ):
        super().__init__()
        self.df = pd.read_csv(data_file, header=None)
        data_file_name = re.findall('/datasets/(.*)-unsupervised',data_file)
        data_file_name = re.sub('-','_',data_file_name[0])
        pickle_file = glob.glob(PROJECT_PATH+"/final_models/train_test/"+data_file_name+"*pkl")
        #self.norm = pd.read_pickle('/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/final_models/train_test/breast_cancer_train.pkl')
        self.norm = pd.read_pickle(pickle_file[0])
        self.row_size = self.df.shape[1] - 1
        self.num_of_rows = self.df.shape[0]
        self.X_orig = self.df.iloc[:,:-1]
        self.X = self.normalize_data_pkl(self.X_orig)
        self.t = self.df.iloc[:,-1]
        # Train data
        self.df_train = pd.read_csv(train_data_file, header=None)
        self.X_train = self.df_train.iloc[:,:-1]
        self.t_train = self.df_train.iloc[:,-1]
        self.distances = self.knn(knn_small=1,knn_high=51)
        self.lof = self.lof(knn_small=10,knn_high=15)
        self.isolation = self.isolation()
        print("hi")

    def __len__(self):
        return th.tensor(len(self.X))

    def __getitem__(self, idx):
        return th.tensor(self.X.iloc[idx].values),idx

    def normalize_data(self, X):
        max_per_col, min_per_col = X.max(), X.min()
        # X = 2 * (X - min_per_col) / (max_per_col - min_per_col) - 1
        if max_per_col == min_per_col:
            return np.zeros_like(X)
        X = (X - min_per_col) / (max_per_col - min_per_col)
        return X

    def normalize_data_pkl(self, X):
        max_per_col, min_per_col = self.norm["max_per_col"], self.norm["min_per_col"]
        X = 2*(X - min_per_col)/(max_per_col-min_per_col) - 1
        return X

    def get_labels(self,idx):
        return self.t[idx]

    def get_all_labels(self):
        return self.t

    def get_row_size(self):
        return self.row_size

    def get_num_of_rows(self):
        return self.num_of_rows

    def get_distances(self):
        return self.distances

    def get_lof(self):
        return self.lof

    def get_isolation(self):
        return self.isolation

    def knn(self, knn_small, knn_high):
        X = self.X
        nbrs = NearestNeighbors(n_neighbors=knn_high)
        nbrs.fit(X)
        distances, indexes = nbrs.kneighbors(X)
        distances_mean = []
        count = 0
        for i in range(knn_small, knn_high):
            print("knn iteration: {}".format(i))
            cur_distances = np.mean(distances[:, 1:i + 1], axis=1)
            min_per_col, max_per_col = np.min(cur_distances), np.max(cur_distances)
            cur_distances = (cur_distances - min_per_col) / (max_per_col - min_per_col)
            distances_mean.append(cur_distances)
            count += 1

        return distances_mean

    def lof(self, knn_small, knn_high):
        lof_list = []
        for i in range(knn_small, knn_high):
            print("LOF iteration: {}".format(i))
            clf = LocalOutlierFactor(n_neighbors=i, novelty=True).fit(self.X_train)
            scores_test = self.normalize_data(-clf.score_samples(self.X_orig))
            lof_list.append(scores_test)

        return lof_list

    def isolation(self):
        isolation_list = []
        for i in range(0, 1):
            print("isolation iteration: {}".format(i))
            clf = IsolationForest(random_state=0).fit(self.X_train)
            scores_test = self.normalize_data(-clf.score_samples(self.X_orig))
            isolation_list.append(scores_test)

        return isolation_list
