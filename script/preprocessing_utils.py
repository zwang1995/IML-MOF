# -*- coding: utf-8 -*-
# @Time:     11/1/2021 10:28 AM
# @Author:   Zihao Wang, zwang@mpi-magdeburg.mpg.de
# @File:     preprocessing_utils.py


import pandas as pd
import numpy as np


def get_FP(items, FP_type):
    if FP_type == "pubchem":
        FPs = np.array([item.pubchem for item in items])
    if FP_type == "maccs":
        FPs = np.array([item.maccs for item in items])
    if FP_type == "jarvis":
        FPs = np.array([item.jarvis for item in items])
    return FPs


def get_prop(items, prop_type):
    if prop_type == "S_label":
        return np.array([item.S_label for item in items]).reshape(-1, 1)
    if prop_type == "S":
        return np.array([item.S for item in items]).reshape(-1, 1)


def get_X_y(pairs):
    X = np.array([np.array(pair[0]) for pair in pairs])
    y = np.array([np.array(pair[1][0]) for pair in pairs])
    name = [pair[2] for pair in pairs]
    return X, y, name


class FP_enc(object):
    def __init__(self, params, name, FPs):
        self.name = name
        self.estate, self.maccs, self.pubchem, self.sub, self.subc = FPs


def assign_PROP(params, items):
    df = pd.read_csv(params["input_csv"])
    MOFs = df[params["mof"]].values
    reg_props = df[params["reg_props"]].values

    MOF2prop = {}
    for i in range(len(df)):
        mof, prop = MOFs[i][0], reg_props[i]
        MOF2prop[mof] = prop

    new_items = []
    for item in items:
        filename = item.name
        if filename in MOFs:
            item.__setattr__("S_label", get_label(params["S_level"], MOF2prop[filename][0]))
            new_items.append(item)
    return new_items


def get_label(levels, value):
    label = None
    level_num = len(levels) + 1
    labels = np.arange(level_num)
    for i in range(level_num - 1):
        if value < levels[i]:
            label = labels[i]
            break
        else:
            i += 1
    if label is None:
        label = labels[-1]
    return label
