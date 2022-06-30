# -*- coding: utf-8 -*-
# @Time:     11/2/2021 10:40 AM
# @Author:   Zihao Wang, zwang@mpi-magdeburg.mpg.de
# @File:     config.py

import numpy as np


def get_params():
    params = {}

    params["in_path"] = "../data/"
    params["out_path"] = "../output/"

    params["input_csv"] = params["in_path"] + "hMOF_list.csv"
    params["txt_file"] = params["out_path"] + "used_items.txt"
    params["his_file"] = params["out_path"] + "train_history.csv"
    params["pred_file"] = params["out_path"] + "predictions.csv"

    params["prepro"] = False
    params["training"] = not params["prepro"]
    params["resample"] = True

    params["mof"] = ["Framework"]
    params["reg_props"] = ["S1"]
    params["rand_seeds"] = [0]

    params["S_levels"] = [[1]]

    return params


def update_params(params, task, FP):
    params["his_file"] = params["out_path"] + "_".join(["train_history", str(task), str(FP)]) + ".csv"
    return params
