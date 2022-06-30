# -*- coding: utf-8 -*-
# @Time:     10/26/2021 8:10 PM
# @Author:   Zihao Wang, zwang@mpi-magdeburg.mpg.de
# @File:     training_utils.py

import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_split(pairs, rand_seed, bs):
    pairs_train, pairs_test = train_test_split(pairs, test_size=0.2, random_state=rand_seed)

    geom_train = np.array([np.array(pair[2]) for pair in pairs_train])
    scaler = StandardScaler()
    scaler.fit(geom_train)

    pairs_valid, pairs_test = train_test_split(pairs_test, test_size=0.5, random_state=rand_seed)
    train_loader = DataLoader(pairs_train, batch_size=bs, shuffle=False)
    valid_loader = DataLoader(pairs_valid, batch_size=len(pairs_valid))
    test_loader = DataLoader(pairs_test, batch_size=len(pairs_test))

    return train_loader, valid_loader, test_loader, scaler


class MOF_Dataset(Dataset):
    def __init__(self, Xs, ys, names):
        self.Xs = Xs
        self.ys = ys
        self.names = names

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, index):
        X = self.Xs[index]
        y = self.ys[index]
        name = self.names[index]
        return X, y, name
