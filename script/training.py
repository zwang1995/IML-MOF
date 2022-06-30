# -*- coding: utf-8 -*-
# @Time:     10/25/2021 11:36 AM
# @Author:   Zihao Wang, zwang@mpi-magdeburg.mpg.de
# @File:     training.py

import warnings

warnings.filterwarnings('ignore')

import csv
import joblib
import pickle
from sklearn.metrics import accuracy_score, f1_score
import torch
from config import *
from preprocessing_utils import *
from training_utils import *

from sklearn.ensemble import RandomForestClassifier


def train(params, items, FP_type, prop_type, model_save=False):
    FPs = get_FP(items, FP_type)
    props = get_prop(items, prop_type)
    names = [item.name for item in items]
    pairs = MOF_Dataset(FPs, props, names)
    pairs_train, pairs_test = train_test_split(pairs,
                                               test_size=0.2,
                                               random_state=params["rand_seed"],
                                               stratify=pairs.ys)
    pairs_valid, pairs_test = train_test_split(pairs_test,
                                               test_size=0.5,
                                               random_state=params["rand_seed"],
                                               stratify=[pair[1][0] for pair in pairs_test])

    X_train, y_train, name_train = get_X_y(pairs_train)

    n_es = [20, 30, 40, 50, 60]
    max_depths = [16, 20, 24, 28, 32]
    min_sam_leafs = [2, 4, 6, 8, 10]
    times = [2, 3, 4, 5, 6]
    if model_save:
        n_es, max_depths, min_sam_leafs, times = params["optimal_hyper"]
    for n_e in n_es:
        for max_depth in max_depths:
            for min_sam_leaf in min_sam_leafs:
                for time in times:

                    model = RandomForestClassifier(n_estimators=n_e,
                                                   max_depth=max_depth,
                                                   min_samples_split=min_sam_leaf * time,
                                                   min_samples_leaf=min_sam_leaf,
                                                   random_state=params["rand_seed"],
                                                   )
                    new_hyper = [n_e, max_depth, min_sam_leaf, time]
                    model.fit(X_train, y_train)

                    def prediction(pair, set):
                        X, y, name = get_X_y(pair)
                        f = model.predict(X)
                        probs = model.predict_proba(X)
                        probs = [str(prob) for prob in probs]
                        # print(probs)
                        sets = [set] * len(name)
                        DF = pd.DataFrame(name, columns=["name"])
                        DF["y"] = y
                        DF["f"] = f
                        DF["set"] = sets
                        DF["prob"] = probs
                        y, f = DF["y"].values, DF["f"].values
                        accu = accuracy_score(y, f)
                        f1 = f1_score(y, f)
                        return round(accu, 4), round(f1, 4), DF

                    accu_tr, f1_tr, df_tr = prediction(pairs_train, "train")
                    accu_val, f1_val, df_val = prediction(pairs_valid, "valid")
                    accu_te, f1_te, df_te = prediction(pairs_test, "test")

                    if prop_type == "S_label":
                        level = params["S_level"]
                    hyper = [level, params["rand_seed"], "RF", FP_type, prop_type]
                    update = ["{:.4f}".format(i) for i in [accu_tr, accu_val, accu_te]] + \
                             ["{:.4f}".format(i) for i in [f1_tr, f1_val, f1_te]]
                    print(new_hyper, " & ".join([str(i) for i in hyper]), ":", " / ".join(update), flush=True)
                    if model_save:
                        new_hyper = list(map(str, new_hyper))
                        np.save(params["out_path"] + "_".join(["single_X_train", prop_type, FP_type]) + ".npy", X_train)

                        df_all = pd.concat([df_tr, df_val, df_te], ignore_index=True)
                        df_all.to_csv(
                            params["out_path"] + "_".join(["single_pred", prop_type, FP_type] + new_hyper) + ".csv")
                        joblib.dump(model, params["out_path"] + "_".join(["single_model", prop_type, FP_type]) + ".pkl")

                        id = open(params["single_his_file"], "w", newline="")
                        writer = csv.writer(id)
                        writer.writerow(new_hyper + hyper + update)
                        id.close()

                    else:
                        id = open(params["his_file"], "a", newline="")
                        writer = csv.writer(id)
                        writer.writerow(new_hyper + hyper + update)
                        id.close()


def training(params, FP_type, prop_type, model_save=False):
    initial = True
    for rs in params["rand_seeds"]:
        params["rand_seed"] = rs
        torch.manual_seed(params["rand_seed"])
        np.random.seed(params["rand_seed"])

        levels = params["S_levels"]
        for level in levels:
            params["S_level"] = level

            print("Params:", params, flush=True)
            filehandler = open(params["in_path"] + "items_8800_comb.pickle", 'rb')
            items = pickle.load(filehandler)
            items = assign_PROP(params, items)

            if initial:
                print("Data points:", len(items), flush=True)
                print(" * S Dist:", np.unique([item.S_label for item in items], return_counts=True), flush=True)
                attrs = list(vars(items[0]).keys())
                print("Items attributions:", attrs, flush=True)
                print("\nStrat training...", flush=True)
                initial = False
            train(params, items, FP_type, prop_type, model_save)


for prop_type in ["S_label"]:
    for FP_type in ["jarvis", "maccs", "pubchem"]:
        params = get_params()
        params = update_params(params, prop_type, FP_type)
        id = open(params["his_file"], "w", newline="")
        id.close()
        training(params, FP_type, prop_type)
