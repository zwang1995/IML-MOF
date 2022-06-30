# -*- coding: utf-8 -*-
# @Time:     12/2/2021 3:34 PM
# @Author:   Zihao Wang, zwang@mpi-magdeburg.mpg.de
# @File:     viz.py


from config import *
from preprocessing_utils import *
from training_utils import *
import pickle, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = "9"  # 10
plt.rcParams["font.family"] = "arial"
plt.rcParams["figure.figsize"] = (3.3, 3)

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from matminer.featurizers.structure import JarvisCFID

import seaborn as sns
import shap

ef = JarvisCFID()
CFID_labels = ef.labels
CFID_labels = [i[4:] for i in CFID_labels]
print(CFID_labels)
s = 80
alpha = 0.8

for FP_type in ["pubchem"]:
    print(FP_type)
    prop_type = "S_label"
    params = get_params()
    params = update_params(params, prop_type, FP_type)
    params["rand_seed"] = 0
    filehandler = open(params["in_path"] + "items_8800_comb.pickle", 'rb')
    items = pickle.load(filehandler)
    items = assign_PROP(params, items)


    def save_FP():
        if FP_type == "jarvis":
            jarvises = []
            for item in items:
                jarvises.append([item.name] + list(item.jarvis))
            df = pd.DataFrame(jarvises, columns=["name"] + CFID_labels)
            print(df.head(5))
            df.to_csv("../data/jarvis.csv")
            print(len(items[0].jarvis))
    # save_FP()

    FPs = get_FP(items, FP_type)
    print(FPs.shape)
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

    X, y, names = get_X_y(pairs)
    if FP_type == "jarvis":
        labels = CFID_labels
    elif FP_type == "pubchem":
        Pub_labels = ["bit_" + str(i) for i in np.arange(X.shape[1])]
        labels = Pub_labels


    model = joblib.load("../output/single_model_S_label_" + FP_type + ".pkl")


    def prediction(pair, set):
        X, y, name = get_X_y(pair)
        f = model.predict(X)
        probs = model.predict_proba(X)
        probs = [str(prob) for prob in probs]
        sets = [set] * len(name)
        DF = pd.DataFrame(name, columns=["name"])
        DF["y"] = y
        DF["f"] = f
        DF["set"] = sets
        DF["prob"] = probs
        return DF
    # df_tr = prediction(pairs_train, "train")
    # df_val = prediction(pairs_valid, "valid")
    # df_te = prediction(pairs_test, "test")
    # df_all = pd.concat([df_tr, df_val, df_te], ignore_index=True)
    # df_all.to_csv("../viz/" + FP_type + ".csv")

    def con_matrix(pairs, set):
        X, y, _ = get_X_y(pairs)
        plt.clf()

        f = model.predict(X)
        con_mat = confusion_matrix(y, f)
        print(con_mat, np.trace(con_mat) / np.sum(con_mat), f1_score(y, f), precision_score(y, f), recall_score(y, f))
        labels = ["C$_2$H$_4$-selective", "C$_2$H$_6$-selective"]
        ax = sns.heatmap(con_mat, linewidths=1, annot=True, fmt="d", cmap="YlGnBu", cbar=False, xticklabels=labels,
                         yticklabels=labels)
        # cbar = ax.collections[0].colorbar
        # cbar.ax.tick_params(labelsize=10)
        plt.xlabel("Predicted label", size=12)
        plt.ylabel("True label", size=12)
        plt.savefig("../viz/ConMat_" + FP_type + "_" + set, dpi=300, bbox_inches="tight", transparent=True)
    # con_matrix(pairs_train, "train")
    # con_matrix(pairs_valid, "valid")
    # con_matrix(pairs_test, "test")

    def shap_global(max_display=10):
        plt.clf()
        X, y, names = get_X_y(pairs_train)
        df2 = pd.DataFrame(X)
        df2.to_csv("../viz/X.csv")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        print(shap_values)
        df2 = pd.DataFrame(shap_values[1])
        df2.to_csv("../viz/df2.csv")
        df2 = pd.DataFrame(np.abs(shap_values[1]))
        df2.to_csv("../viz/df2_abs.csv")
        print(shap_values[0].shape, shap_values[1].shape)
        shap.summary_plot(shap_values[1],
                          features=X,
                          feature_names=labels,
                          max_display=max_display,
                          plot_type="bar", plot_size=(2.5, 6), color="#469990", show=False)
        if FP_type == "jarvis":
            plt.xlim([0, 0.03])
        elif FP_type == "pubchem":
            plt.xlim([0, 0.03])
        plt.tick_params(axis="x", direction="in")
        plt.tick_params(axis="y", direction="in")
        plt.xlabel("mean(|SHAP value|)", size=12)
        for pos in ["right", "top", "bottom", "left"]:
            plt.gca().spines[pos].set_visible(True)
        plt.gca().spines["right"].set_color("lightgray")
        plt.savefig("../viz/SHAP_left_" + FP_type + "_" + str(max_display), dpi=300, bbox_inches="tight",
                    transparent=True)

        plt.clf()
        shap.summary_plot(shap_values[1],
                          features=X,
                          feature_names=labels,
                          max_display=max_display,
                          plot_size=(7.5, 6),
                          show=False)
        if FP_type == "jarvis":
            plt.xlim([-0.105, 0.065])
        elif FP_type == "pubchem":
            pass
            plt.xlim([-0.105, 0.065])
        plt.tick_params(axis="x", direction="in")
        plt.tick_params(axis="y", direction="in")
        plt.xlabel("SHAP value", size=12)
        for pos in ["right", "top", "bottom", "left"]:
            plt.gca().spines[pos].set_visible(True)
        plt.gca().spines["left"].set_color("lightgray")
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.savefig("../viz/SHAP_right_" + FP_type + "_" + str(max_display), dpi=300, bbox_inches="tight",
                    transparent=True)
    # shap_global(15)

    def shap_global_metal():
        metal_index = [55, 61, 62, 71]
        plt.clf()
        X, y, names = get_X_y(pairs_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        print(shap_values)
        df2 = pd.DataFrame(shap_values[1])
        df2.to_csv("../viz/df2.csv")
        df2 = pd.DataFrame(np.abs(shap_values[1]))
        df2.to_csv("../viz/df2_abs.csv")
        print(shap_values[0].shape, shap_values[1].shape)
        shap.summary_plot(shap_values[1][:, metal_index],
                          features=X[:, metal_index],
                          feature_names=[labels[i] for i in metal_index],
                          plot_type="bar", plot_size=(2.5, 2), color="#469990", show=False)
        if FP_type == "jarvis":
            plt.xlim([0, 0.01])
        elif FP_type == "pubchem":
            plt.xlim([0, 0.006])
            plt.xticks([0, 0.002, 0.004, 0.006])
        plt.tick_params(axis="x", direction="in")
        plt.tick_params(axis="y", direction="in")
        plt.xlabel("mean(|SHAP value|)", size=12)
        for pos in ["right", "top", "bottom", "left"]:
            plt.gca().spines[pos].set_visible(True)
        plt.gca().spines["right"].set_color("lightgray")
        plt.savefig("../viz/SHAP_left_" + FP_type + "_metal", dpi=300, bbox_inches="tight", transparent=True)

        plt.clf()
        shap.summary_plot(shap_values[1][:, metal_index],
                          features=X[:, metal_index],
                          feature_names=[labels[i] for i in metal_index],
                          plot_size=(5.5, 2),
                          show=False)
        if FP_type == "jarvis":
            plt.xlim([-0.045, 0.045])
        elif FP_type == "pubchem":
            pass
            plt.xlim([-0.045, 0.045])
        plt.tick_params(axis="x", direction="in")
        plt.tick_params(axis="y", direction="in")
        plt.xlabel("SHAP value", size=12)
        for pos in ["right", "top", "bottom", "left"]:
            plt.gca().spines[pos].set_visible(True)
        plt.gca().spines["left"].set_color("lightgray")
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.savefig("../viz/SHAP_right_" + FP_type + "_metal", dpi=300, bbox_inches="tight", transparent=True)
    # shap_global_metal()

    def shap_local():
        plt.rcParams["font.size"] = "12"
        plt.clf()
        X, y, names = get_X_y(pairs_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        print(explainer.expected_value)

        main_features = np.abs(shap_values[1]).mean(0).argsort()[-10:]

        for mof in ["hMOF-1002807", "hMOF-6004362"]:
            for num, name in enumerate(names):
                if name == mof:
                    print(num, name, names[num], y[num])
                    print([i for i in X[num, :]])
                    # shap.force_plot(explainer.expected_value[1],
                    #                 shap_values[1][num,:],
                    #                 np.around(X[num,:], decimals=2),
                    #                 feature_names=labels,
                    #                 matplotlib=True, show=False, figsize=(10,3))
                    # plt.savefig("../viz/SHAP_forceplot_" + FP_type + "_" + name, dpi=300, bbox_inches="tight", transparent=True)

                    plt.clf()
                    # explainer = shap.Explainer(model)
                    # shap_values = explainer(X)
                    # print(shap_values)
                    # shap.plots.waterfall(shap_values[0][0])

                    shap.plots.waterfall(shap.Explanation(values=shap_values[1][num],
                                                          base_values=explainer.expected_value[1], data=X[num, :],
                                                          feature_names=labels), show=False)
                    plt.xlim([0.7, 1])
                    plt.savefig("../viz/SHAP_waterfall_" + FP_type + "_" + name, dpi=300, bbox_inches="tight",
                                transparent=True)
    # shap_local()

""" Figure 2 """
def value_dist():
    df = pd.read_csv("../input/hMOF_list.csv")
    s0 = df[df.S1 < 1]["S1"].values
    s1 = df[df.S1 > 1]["S1"].values
    print(len(s0), len(s1))
    plt.clf()
    # plt.rcParams['patch.linewidth'] = 0.5
    plt.rcParams['patch.edgecolor'] = 'none'
    sns.histplot(s0, color='tab:blue')
    sns.histplot(s1, color='tab:green')

    plt.xlim([-0.1, 4.1])
    plt.ylim([0, 400])
    plt.xlabel("Selectivity value", size=10)
    plt.ylabel("Count", size=10)
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    plt.legend(labels=["C$_2$H$_4$-selective", "C$_2$H$_6$-selective"], prop={'size': 8})
    # plt.axvline(x=1, c="tab:red", lw=0.7)
    plt.savefig("../viz/Sele_values", dpi=300, bbox_inches="tight", transparent=True)
# value_dist()

""" Figure 5 """
def fea_selec():
    candidates = np.array([21164,
                           12767, 12765, 12765, 6579, 5809,
                           3972, 3972, 2179, 2179, 2179,
                           1986, 1439, 1439, 1439, 583,
                           371, 371, 36, 36, 36]) / 1000
    feature_nums = np.arange(len(candidates))
    plt.plot(feature_nums, candidates, '-o', markersize=4, lw=1)
    plt.xlim([-0.5, 20.5])
    plt.ylim([0, 25])
    plt.xlabel("Number of features", size=10)
    plt.ylabel("Number of candidates (×10$^3$)", size=10)
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    plt.axhline(y=candidates[0] * 0.05, c="tab:red", ls="--", lw=1)
    plt.savefig("../viz/Feature_Candidate", dpi=300, bbox_inches="tight", transparent=True)
# fea_selec()

""" Figure 6 """
def gcmc_viz():
    plt.rcParams["figure.figsize"] = (8, 3)
    df = pd.read_csv("../viz/data/gcmc_569.csv")
    index, S = df.COUNT.values, df.S.values
    print("total number:", len(S))
    index_4, index_6, S_4, S_6 = [], [], [], []
    for i, s in zip(index, S):
        if s > 1:
            index_6.append(i)
            S_6.append(s)
        elif s < 1:
            index_4.append(i)
            S_4.append(s)
    print("C2H4 number:", len(S_4))
    print("C2H6 number:", len(S_6))
    print("nan number:", len(S) - len(S_4) - len(S_6))
    plt.axhline(y=1, color="k", ls='--', lw=0.8, zorder=0)
    plt.scatter(index_4, S_4, s=20, c="grey", zorder=1)
    plt.scatter(index_6, S_6, s=20, c="tab:green", zorder=2)
    plt.xlim([-5, 605])
    plt.ylim([-0.1, 7.1])
    plt.xlabel("Index of the MOF candidate", size=10)
    plt.ylabel("C$_2$H$_6$/C$_2$H$_4$ selectivity", size=10)
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    plt.savefig("../viz/gcmc_viz", dpi=300, bbox_inches="tight", transparent=True)
# gcmc_viz()

""" Figure S3 """
def probability_vs_selectivity():
    df = pd.read_csv("../viz/data/probability vs selectivity.csv", index_col=0)
    df = df[df["set"] == "train"]
    print(df)
    x = df["jarvis_prob"].values
    y = df["selectivity"].values
    plt.scatter(x, y, s=10)
    plt.xlim([-0.05, 1.05])
    plt.xlabel("C$_2$H$_6$-selective probability ", size=12)
    plt.ylabel("C$_2$H$_6$/C$_2$H$_4$ selectivity", size=12)
    plt.savefig("../viz/prob_vs_sele", dpi=300, bbox_inches="tight", transparent=True)
# probability_vs_selectivity()
