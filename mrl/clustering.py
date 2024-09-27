# -*- coding: utf-8 -*-
"""
This file contains the functions to generate and perform the MDP clustering

Created on Sun Mar  1 18:48:20 2020

@author: omars
"""

# Load Libraries
from typing import Tuple, List, Dict, Union
import gc
import os
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import time
from tqdm import tqdm  # progress bar
import binascii
from copy import deepcopy
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from collections import Counter
from itertools import groupby
from operator import itemgetter
import sys

from testing import (
    training_value_error,
    training_accuracy,
    predict_cluster,
    R2_value_testing,
    testing_value_error,
    testing_accuracy,
    next_clusters,
)

# suppressing warnings from here because sklearn forces warnings
# https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

# Funtions for Initialization


def split_train_test_by_id(data, test_ratio, id_column):
    """
    Splits the data into training and testing sets based on ID.

    Parameters:
    data (DataFrame): The input dataframe containing all the data.
    test_ratio (float): The portion of data to be used for testing.
    id_column (str): The name of the identifying ID column.

    Returns:
    tuple: Training and testing dataframes.
    """

    def test_set_check(identifier, test_ratio):
        return binascii.crc32(np.int64(identifier)) & 0xFFFFFFFF < test_ratio * 2 ** 32

    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def initializeClusters(
    df,
    clustering="Agglomerative",
    n_clusters=None,
    distance_threshold=0.3,
    random_state=0,
    verbose=False,
):
    """
    Initializes clusters in the dataframe using the specified clustering algorithm.

    Parameters:
    df (DataFrame): Input dataframe, must contain a "RISK" column.
    clustering (str): Clustering algorithm ('KMeans', 'Agglomerative', 'Birch').
    n_clusters (int): Number of clusters (for KMeans and Birch).
    distance_threshold (float): Clustering diameter for Agglomerative clustering.
    random_state (int): Random seed for the clustering.
    verbose (bool): Verbosity flag.

    Returns:
    DataFrame: DataFrame with 'CLUSTER' and 'NEXT_CLUSTER' columns.

    Code Update note [#DEBUG]: Filtered pathological cases of 1 data point 
    and all data points having the same risk.
    """
    # df = df.copy() saves a little bit of memory
    # we're separating here because we don't want end states to mix with usual states
    usual_obs = df["ACTION"] != "None"
    end_obs = (
        df["ACTION"] == "None"
    )  # observations where we're in an 'end state', i.e., a state which marks the end of the game
    cluster_index_begin = 0
    for obs_type in [usual_obs, end_obs]:
        if verbose:
            print("begin clustering rewards using method: ", clustering)

        risk_array = np.array(df[obs_type].RISK).reshape(-1, 1) #Risks of data points, to cluster
        #First, some pathological cases
        if risk_array.size <=1: #Case of only 1 data point, nothing to cluster
            output = np.array([0])
        elif np.all(risk_array == risk_array[0]): #Case all points have the same risk, nothing to cluster
            output = np.array([0]*risk_array.size)
        else:
            if clustering == "KMeans":
                output = (
                    KMeans(n_clusters=n_clusters, random_state=random_state)
                    .fit(risk_array)
                    .labels_
                )
                output = LabelEncoder().fit_transform(
                    output
                )  # relabel to remove empty clusters
            elif clustering == "Agglomerative":  # not feasible O(ds^2) time, memory
                output = (
                    AgglomerativeClustering(
                        n_clusters=n_clusters, distance_threshold=distance_threshold
                    )
                    .fit(risk_array)
                    .labels_
                )
            elif clustering == "Birch":
                output = (
                    Birch(n_clusters=n_clusters)
                    .fit(risk_array)
                    .labels_
                )
            else:
                output = LabelEncoder().fit_transform(
                    risk_array
                )
        if verbose:
            print("finish clustering")
        df.loc[obs_type, "CLUSTER"] = (
            output + cluster_index_begin
        )  # enforce separate clusters for end and non-end states
        cluster_index_begin += int(max(output)) + 1
    df["CLUSTER"] = df["CLUSTER"].astype(int)
    df["NEXT_CLUSTER"] = df["CLUSTER"].shift(-1)
    df.loc[df["ID"] != df["ID"].shift(-1), "NEXT_CLUSTER"] = 0
    df["NEXT_CLUSTER"] = df["NEXT_CLUSTER"].astype(int)
    df.loc[df["ID"] != df["ID"].shift(-1), "NEXT_CLUSTER"] = "None"

    # change end state to 'end'
    # so here end state is when no action is taken. Otherwise the next state is 'None'
    df.loc[df["ACTION"] == "None", "NEXT_CLUSTER"] = "End"

    return df


# Function for the Iterations


def findContradictionStochastic(df, th, p_feats):
    """Similar to below. Stochastic version. p_feats is number of features"""
    # filter out relevant entries in datasets
    X = df[df.NEXT_CLUSTER != "None"]

    def next_cluster_std_weighted(df):
        """
        Given a cluster-action pair, check how "uniform" the distribution of next cluster is.
        """
        # Step 1. Use regression to smooth the observations
        X_regr = df.iloc[:, 2 : p_feats + 2]
        y_regr_raw = df["NEXT_CLUSTER"]
        encoder = LabelEncoder()
        y_regr = encoder.fit_transform(y_regr_raw)
        mlp = MLPClassifier((10,), "relu", alpha=0.05, learning_rate_init=0.2, tol=1e-3)
        mlp.fit(X_regr, y_regr)

        # Step 2. The variances in the predicted probabilities is used to measure how much variation the cluster has
        y_preds = mlp.predict_proba(X_regr)
        return sum(np.std(y_preds, axis=0) * np.sum(y_regr, axis=0))

    # Step 3. Find the cluster-action pair with highest variance. Early stop if variance less than threshold.
    stds = X.groupby(["CLUSTER", "ACTION"]).apply(next_cluster_std_weighted)

    if stds.max() < th:
        return (-1, -1)
    return stds.idxmax()


def findContradiction(df, th, verbose=False):
    """
    Finds the initial cluster and action that have the most number of contradictions.

    Parameters:
    df (DataFrame): Input dataframe.
    th (int): Threshold split size.
    verbose (bool): Verbosity flag.

    Returns:
    tuple: (initial cluster, action) with the most contradictions or (-1, -1) if none found.
    """

    st = time.time()
    X = df.loc[:, ["CLUSTER", "NEXT_CLUSTER", "ACTION"]]
    X = X[X.NEXT_CLUSTER != "None"]
    count = X.groupby(["CLUSTER", "ACTION"])["NEXT_CLUSTER"].nunique()
    contradictions = list(count[list(count > 1)].index)
    if verbose:
        print(f"Find Contradiction Step 1 Complete in {time.time() - st}")

    """Might rewrite to get O(dataset size)"""
    if not len(contradictions):
        return (-1, -1)

    st = time.time()
    transition_counts = (
        X.groupby(["CLUSTER", "ACTION", "NEXT_CLUSTER"]).size().unstack("NEXT_CLUSTER")
    )
    transition_counts = transition_counts.loc[contradictions]

    def second_largest(series):
        ssort = series.sort_values(ascending=False)
        return ssort.iloc[1]

    seclargest = transition_counts.apply(second_largest, axis=1)
    c, a = seclargest.idxmax()
    if verbose:
        print(f"Find Contradiction my method finished in {time.time()-st}")
    if seclargest.max() > th:
        return (c, a)
    return (-1, -1)


def contradiction(df, i, a):
    """
    Outputs one found contradiction given a dataframe, a cluster, and an action.

    Parameters:
    df (DataFrame): Input dataframe.
    i (int): Initial cluster.
    a (int): Action taken.

    Returns:
    tuple: (action, most common NEXT_CLUSTER) or (None, None) if none found.
    """

    nc = list(
        df.query("CLUSTER == @i")
        .query("ACTION == @a")
        .query('NEXT_CLUSTER != "None"')["NEXT_CLUSTER"]
    )
    if len(nc) == 1:
        return (None, None)
    else:
        return a, multimode(nc)[0]


def multimode(data):
    """
    Returns a list of the most frequently occurring values.

    Parameters:
    data (iterable): Input data.

    Returns:
    list: Most frequently occurring values.
    """
    counts = Counter(iter(data)).most_common()
    maxcount, mode_items = next(groupby(counts, key=itemgetter(1)), (0, []))
    return list(map(itemgetter(0), mode_items))


def parse_classifier(
    classification, split_classifier_params  # string: classification aglo
):
    """
    Get classifier based on the specified type.

    Parameters:
    classification (str): Classification algorithm.
    split_classifier_params (dict): Classifier parameters.

    Returns:
    Classifier: Initialized classifier.
    """

    if classification == "LogisticRegression":
        return LogisticRegression(**split_classifier_params)
    if classification == "LogisticRegressionCV":
        return LogisticRegressionCV(**split_classifier_params)
    if classification == "DecisionTreeClassifier":
        return DecisionTreeClassifier(**split_classifier_params)

    if classification == "RandomForestClassifier":
        return RandomForestClassifier(**split_classifier_params)

    if classification == "MLPClassifier":
        return MLPClassifier(**split_classifier_params)
    if classification == "AdaBoostClassifier":
        return AdaBoostClassifier(**split_classifier_params)
    raise ValueError("Incorrect Classifier Type")


def splitStochastic(
    df, i, a, p_feats, k, nsplits, classification, split_classifier_params
):
    """
    Performs stochastic splitting of the initial cluster.

    Parameters:
    df (DataFrame): Input dataframe.
    i (int): Initial cluster.
    a (int): Action taken.
    p_feats (int): Number of features.
    k (int): Indexer for next cluster.
    nsplits (int): Number of splits of the initial cluster.
    classification (str): Classification algorithm.
    split_classifier_params (dict): Classifier parameters.

    Returns:
    DataFrame: Updated dataframe after splitting.
    """

    X = df[(df["CLUSTER"] == i) & (df["ACTION"] == a)]
    X = X[X["NEXT_CLUSTER"] != "None"]
    unlabeled_part = df[df["CLUSTER"] == i]

    # Step 1. Use regression to smooth the observations
    X_regr = X.iloc[:, 2 : p_feats + 2]
    y_regr_raw = X["NEXT_CLUSTER"]
    encoder = LabelEncoder()
    y_regr = encoder.fit_transform(y_regr_raw)
    scaler = MinMaxScaler()
    X_regr_normalized = scaler.fit_transform(X_regr)
    regressor = DecisionTreeClassifier()
    regressor.fit(X_regr_normalized, y_regr)
    y_preds = regressor.predict_proba(X_regr_normalized)

    # Step 2. Use K-Means Clustering to split the predicted probabilities to find ideal split groups
    clusterer = KMeans(n_clusters=nsplits)
    target_groups = clusterer.fit_predict(y_preds)
    X.insert(X.shape[1], "LABEL", target_groups)

    # Step 3. Calculate intercluster standard deviations
    weights = np.bincount(y_regr)
    subcluster_stds = []
    for i_subcluster in range(nsplits):
        subcluster_stds.append(
            (y_preds[target_groups == i_subcluster].std(axis=0) * weights).sum()
        )
    # print("subcluster stds", subcluster_stds)

    # print("Saving presmooth labels")

    # Step 4. Reassign clusters
    m = parse_classifier(classification, split_classifier_params)
    df, score = split_postlabel(df, X, unlabeled_part, m, nsplits, k, p_feats, i)
    return df


def split(
    df,
    i,
    a,
    c,
    pfeatures,
    k,
    classification="LogisticRegression",
    split_classifier_params={"random_state": 0},
):
    """
    Resolves contradictions by splitting the data.

    Parameters:
    df (DataFrame): Input dataframe.
    i (int): Initial cluster.
    a (int): Action taken.
    c (int): Target cluster.
    pfeatures (int): Number of features.
    k (int): Indexer for next cluster.
    classification (str): Classification algorithm.
    split_classifier_params (dict): Classifier parameters.

    Returns:
    tuple: Updated dataframe and best fit score for the splitting model.
    """

    labeled_parts = df[
        (df["CLUSTER"] == i) & (df["ACTION"] == a) & (df["NEXT_CLUSTER"] != "None")
    ]

    """
    unlabeled_parts = df[(df['CLUSTER'] == i) & (
        (df['NEXT_CLUSTER'] == 'None') | (df['ACTION'] != a))]
    """
    unlabeled_parts = df[df["CLUSTER"] == i]

    labeled_parts["LABEL"] = (labeled_parts["NEXT_CLUSTER"] == c).astype(int)

    m = parse_classifier(classification, split_classifier_params)
    df, score = split_postlabel(
        df, labeled_parts, unlabeled_parts, m, 2, k, pfeatures, i
    )
    return df, score


def split_postlabel(
    df,
    labeled_part,
    unlabeled_parts,
    classifier,
    n_classes,
    k,
    p_features,
    init_cluster,
):
    """
    df_cluster: all the datapoints in the cluster to split
    points in labeled_part have some finite list of labels
    returns:
    - post split df with clusters relabeled for points in original split cluster
    - score of classifier which is used to split cluster
    """

    relabel_train = True

    tr_X = labeled_part.iloc[:, 2 : 2 + p_features]
    tr_y = labeled_part["LABEL"]

    classifier.fit(tr_X, tr_y.values.ravel())
    try:
        score = classifier.best_score_
    except:
        score = None

    test_X = unlabeled_parts.iloc[:, 2 : 2 + p_features]

    if len(unlabeled_parts):
        Y = classifier.predict(test_X)
        unlabeled_parts["GROUP"] = Y.ravel()

    # labeled_part.to_pickle("saved_models/df_debug_presmoothsplit.pkl")
    # unlabeled_parts.to_pickle("saved_models/df_debug_postsmoothsplit.pkl")
    # print("Saving presmooth labels")

    for cluster_index in range(1, n_classes):
        ids = labeled_part.loc[labeled_part["LABEL"] == cluster_index].index.values
        if len(unlabeled_parts):
            id2 = unlabeled_parts.loc[
                unlabeled_parts["GROUP"] == cluster_index
            ].index.values
            if relabel_train:
                ids = id2
            else:
                ids = np.concatenate((ids, id2))

        # update the clusters and next_cluster of previous ids
        # print(df.groupby(['CLUSTER', 'NEXT_CLUSTER']).count())
        assert (
            df.loc[df.index.isin(ids), "CLUSTER"] == init_cluster
        ).all(), "trying to reassign cluster to points out of original cluster"
        cluster_update(df, ids, k + cluster_index - 1)
    # print(i,a,c)
    # print(df.groupby(['CLUSTER', 'NEXT_CLUSTER']).count())
    # raise Exception
    return df, score


def intercluster_std(df, p_feats):
    """
    Given a cluster-action pair, check how "uniform" the distribution of next cluster is.
    """
    # Step 1. Use regression to smooth the observations
    X_regr = df.iloc[:, 2 : p_feats + 2]
    y_regr_raw = df["NEXT_CLUSTER"]
    encoder = LabelEncoder()
    y_regr = encoder.fit_transform(y_regr_raw)
    mlp = MLPClassifier((10,), "relu", alpha=0.05, learning_rate_init=0.2, tol=1e-3)
    mlp.fit(X_regr, y_regr)

    # print(X_regr, y_regr)

    # Step 2. The variances in the predicted probabilities is used to measure how much variation the cluster has
    y_preds = mlp.predict_proba(X_regr)
    return sum(np.std(y_preds, axis=0) * np.sum(y_regr, axis=0))


def cluster_update(df, ids, k):
    df.loc[df.index.isin(ids), "CLUSTER"] = k
    previds = ids - 1
    df.loc[
        (df.index.isin(previds)) & (df["ID"] == df["ID"].shift(-1)), "NEXT_CLUSTER"
    ] = k
    return df


def splitter(
    df: pd.DataFrame,
    pfeatures: int,
    th: int,
    eta: int = 25,
    precision_thresh: float = 1e-14,
    df_test: pd.DataFrame = None,
    testing: bool = False,
    max_k: int = 6,
    classification: str = None,
    split_classifier_params: Dict = None,
    h: int = 5,
    gamma: int = 1,
    verbose: bool = False,
    n: int = -1,
    plot: bool = False,
    save_epoch: bool = False,
    save_path: str = None,
    save_every: int = 1,
    eval_samples: int = None,
    stochastic: bool = False,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Union[None, pd.DataFrame],
    pd.DataFrame,
    int,
    List[float],
    Union[None, List[float]],
]:
    """
    Performs the splitting algorithm to resolve contradictions in the dataset.

    Args:
        df (pd.DataFrame): The input dataframe.
        pfeatures (int): Number of features.
        th (int): Threshold for minimum split.
        eta (int, optional): Incoherence threshold for splits. Defaults to 25.
        precision_thresh (float, optional): Precision threshold for new minimum value error. Defaults to 1e-14.
        df_test (pd.DataFrame, optional): Test dataframe for cross-validation. Defaults to None.
        testing (bool, optional): Flag to indicate cross-validation. Defaults to False.
        max_k (int, optional): Maximum number of clusters. Defaults to 6.
        classification (str, optional): Classification algorithm. Defaults to None.
        split_classifier_params (Dict, optional): Classification parameters. Defaults to None.
        h (int, optional): Hyperparameter for training value error calculation. Defaults to 5.
        gamma (int, optional): Hyperparameter for training value error calculation. Defaults to 1.
        verbose (bool, optional): Verbosity flag. Defaults to False.
        n (int, optional): Number of clusters for vertical line in plot. Defaults to -1.
        plot (bool, optional): Flag to plot the results. Defaults to False.
        save_epoch (bool, optional): Flag to save dataframe at each epoch. Defaults to False.
        save_path (str, optional): Path to save the dataframe. Defaults to None.
        save_every (int, optional): Interval for saving epochs. Defaults to 1.
        eval_samples (int, optional): Number of evaluation samples. Defaults to None.
        stochastic (bool, optional): Flag for stochastic processing. Defaults to False.

    Returns:
        Tuple containing:
        - The final resulting dataframe.
        - DataFrame of incoherences.
        - DataFrame of training errors.
        - DataFrame of testing errors or None.
        - DataFrame with the optimal split.
        - Optimal number of clusters.
        - List of split scores.
        - List of training errors or None.
    """
    # initializing lists for error & accuracy data
    testing_R2 = []
    training_acc = []
    testing_acc = []
    testing_error = []
    training_error = []

    incoherences = []
    split_scores = []
    thresholds = []

    # determine if the problem has OG cluster
    if "OG_CLUSTER" in df.columns:
        grid = True
    else:
        grid = False

    k = df["CLUSTER"].nunique()  # initial number of clusters
    nc = k  # number of clusters

    df_new = deepcopy(df)

    # storing optimal df
    best_df = None
    opt_k = None
    min_error = float("inf")

    # backup values in case threshold fails
    backup_min_error = float("inf")
    backup_df = None
    backup_opt_k = None

    # Setting progress bar--------------
    split_bar = tqdm(range(max_k - k))
    split_bar.set_description("Splitting...")

    # Setting progress bar--------------
    for i in split_bar:
        split_bar.set_description("Splitting... |#Clusters:%s" % (nc))
        cont = False
        if verbose:
            print("Finding contradiction...")

        st = time.time()
        if not stochastic:
            c, a = findContradiction(df_new, th)
        else:
            c, a = findContradictionStochastic(df_new, th, pfeatures)

        if verbose:
            print(f"Found contradiction in {time.time()-st}!")

        gc.collect()
        if (
            c != -1
        ):  # this means that the number of contradictions/variance is over the threshold, so no early stopping

            # print("Splitting...")
            st = time.time()

            if not stochastic:
                # finding contradictions and splitting
                a, b = contradiction(df_new, c, a)
                if verbose:
                    print(
                        "Cluster splitted",
                        c,
                        "| Action causing contradiction:",
                        a,
                        "| Cluster most elements went to:",
                        b,
                    )
                df_new, score = split(
                    df_new,
                    c,
                    a,
                    b,
                    pfeatures,
                    nc,
                    classification,
                    split_classifier_params,
                )
                split_scores.append(score)

            else:
                if verbose:
                    print("Cluster splitted", c, "| Action causing contradiction:", a)
                df_new = splitStochastic(
                    df_new,
                    c,
                    a,
                    pfeatures,
                    nc,
                    2,
                    classification,
                    split_classifier_params,
                )

            if verbose:
                print(f"Split clusters in {time.time() - st}!")

            if save_epoch and (
                i % save_every == 0
            ):  # don't want to keep saving huge disk usage
                if verbose:
                    print("Saving checkpoint")
                df_new.to_csv(save_path / f"df_epoch_{i}.csv")

            if verbose:
                print("Calculating Incoherences...")
            # calculate incoherences

            st = time.time()
    
            if not stochastic:
                next_clus = next_clusters(df_new)
                next_clus["incoherence"] = (1 - next_clus["purity"]) * next_clus[
                    "count"
                ]
                next_clus.reset_index(inplace=True)
                next_clus = next_clus.groupby("CLUSTER").sum()
                max_inc = next_clus["incoherence"].max()
                incoherences.append(max_inc)
                if verbose:
                    print(f"Calculated Incoherences in {time.time()-st}")

            # error and accuracy calculations
            st = time.time()
            if verbose:
                print("Calculating Evaluation Metrics...")

            if verbose:
                print("Calculating R2 metric...")
                print("Calculating training error...")

            train_error = training_value_error(
                df_new,
                gamma,
                relative=False,
                h=h,
                eval_samples=eval_samples,
                stochastic=stochastic,
            )
            training_error.append(train_error)
            
            if verbose:
                print("Training Error", train_error)
                print("Calculating training accuracy (grid only) ...")
            
            if grid and not stochastic:
                train_acc = training_accuracy(df_new)[0]
                training_acc.append(train_acc)
            
            if verbose:
                print("Testing...")
            
            if testing and not stochastic:
                model = predict_cluster(df_new, pfeatures)
                R2_test = R2_value_testing(df_test, df_new, model, pfeatures)
                testing_R2.append(R2_test)
                test_error = testing_value_error(
                    df_test, df_new, model, pfeatures, gamma, relative=False, h=h
                )
                testing_error.append(test_error)

                if grid:
                    test_acc = testing_accuracy(df_test, df_new, model, pfeatures)[0]
                    testing_acc.append(test_acc)
            gc.collect()
            
            # printing error and accuracy values
            if verbose:
                # print('training value R2:', R2_train)
                print("training value error:", train_error)
                if grid and not stochastic:
                    print("training accuracy:", train_acc)
                if testing and not stochastic:
                    print("testing value R2:", R2_test)
                    print("testing value error:", test_error)
                    if grid:
                        print("testing accuracy:", test_acc)

            if not stochastic:
                if verbose:
                    print("Calculating Threshold...")
                # update optimal dataframe if inc threshold and min error met
                # threshold calculated using eta * sqrt(number of datapoints) /
                # number of clusters
                threshold = eta * df_new.shape[0] ** 0.5 / (nc + 1)
                thresholds.append(threshold)
                if verbose:
                    print("threshold:", threshold, "max_incoherence:", max_inc)

                # print(f"Evaluation complete in {time.time()-st}")

            if verbose:
                print("Saving best model...")
            st = time.time()
            # only update the best dataframe if training error is smaller
            # than previous training error by at least precision_thresh,
            # and also if maximum incoherence is lower than calculated threshold
            if verbose:
                print("train error: ", train_error)
            if train_error < (min_error - precision_thresh):
                # if max_inc < threshold:
                min_error = train_error
                best_df = df_new.copy()
                opt_k = nc + 1
                if verbose:
                    print("new opt_k", opt_k)

            # code for storing optimal clustering even if incorrect incoherence
            # threshold is chosen and nothing passes threshold; to prevent
            # training interruption
            elif opt_k == None and train_error < (backup_min_error - precision_thresh):
                backup_min_error = train_error
                backup_df = df_new.copy()
                backup_opt_k = nc + 1

            cont = True
            nc += 1
            if verbose:
                print(f"Model saved in {time.time()-st}")
        if not cont:
            break
        if nc >= max_k:
            if verbose:
                print("Optimal # of clusters reached")
            break

    # in the case that threshold prevents any values from passing, use backup
    if opt_k == None:
        opt_k = backup_opt_k
        best_df = backup_df
        min_error = backup_min_error

    # plotting functions
    # Plotting accuracy and value R2
    its = np.arange(k + 1, nc + 1)
    if plot:
        if grid and not stochastic:
            fig1, ax1 = plt.subplots()
            ax1.plot(its, training_acc, label="Training Accuracy")
            if testing:
                ax1.plot(its, testing_acc, label="Testing Accuracy")
            if n > 0:
                ax1.axvline(
                    x=n, linestyle="--", color="r"
                )  # Plotting vertical line at #cluster =n
            ax1.set_ylim(0, 1)
            ax1.set_xlabel("# of Clusters")
            ax1.set_ylabel("R2 or Accuracy %")
            ax1.set_title("R2 and Accuracy During Splitting")
            ax1.legend()
        ## Plotting value error E((v_est - v_true)^2)
        fig2, ax2 = plt.subplots()
        ax2.plot(its, training_error, label="Training Error")
        if testing and not stochastic:
            ax2.plot(its, testing_error, label="Testing Error")
        if n > 0:
            ax2.axvline(
                x=n, linestyle="--", color="r"
            )  # Plotting vertical line at #cluster =n
        ax2.set_ylim(0)
        ax2.set_xlabel("# of Clusters")
        ax2.set_ylabel("Value error")
        ax2.set_title("Value error by number of clusters")
        ax2.legend()
        plt.show()

    df_train_error = pd.DataFrame(
        list(zip(its, training_error)), columns=["Clusters", "Error"]
    )
    df_incoherences = pd.DataFrame(
        list(zip(its, incoherences)), columns=["Clusters", "Incoherences"]
    )
    if testing and not stochastic:
        df_test_error = pd.DataFrame(
            list(zip(its, testing_error)), columns=["Clusters", "Error"]
        )
        return (
            df_new,
            df_incoherences,
            df_train_error,
            df_test_error,
            best_df,
            opt_k,
            split_scores,
            None,
        )
    return (
        df_new,
        df_incoherences,
        df_train_error,
        testing_error,
        best_df,
        opt_k,
        split_scores,
        training_error,
    )


# Splitter algorithm with Group K-fold cross-validation (number of folds from param cv)
# Returns dataframes of incoherences, errors, and splitter split-scores; these
# can be used to determine optimal clustering.
def fit_CV(
    df,
    pfeatures,
    th,
    clustering,
    distance_threshold,
    eta,
    precision_thresh,
    classification,
    split_classifier_params,
    max_k,
    n_clusters,
    random_state,
    h,
    gamma=1,
    verbose=False,
    cv=5,
    n=-1,
    plot=False,
):
    """
    Fits the model using cross-validation.

    Args:
        df (pd.DataFrame): The input dataframe.
        pfeatures (int): Number of features.
        th (int): Threshold for minimum split.
        clustering (str): Clustering algorithm.
        distance_threshold (float): Distance threshold.
        eta (int): Incoherence threshold for splits.
        precision_thresh (float): Precision threshold for new minimum value error.
        classification (str): Classification algorithm.
        split_classifier_params (Dict): Classification parameters.
        max_k (int): Maximum number of clusters.
        n_clusters (int): Initial number of clusters.
        random_state (int): Random state for reproducibility.
        h (int): Hyperparameter for training value error calculation.
        gamma (int, optional): Hyperparameter for training value error calculation. Defaults to 1.
        verbose (bool, optional): Verbosity flag. Defaults to False.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
        n (int, optional): Number of clusters for vertical line in plot. Defaults to -1.
        plot (bool, optional): Flag to plot the results. Defaults to False.

    Returns:
        List[Dict]: List of dictionaries containing results for each fold.
    """
    df_training_error = pd.DataFrame(columns=["Clusters"])
    df_testing_error = pd.DataFrame(columns=["Clusters"])
    df_incoherences = pd.DataFrame(columns=["Clusters"])

    gkf = GroupKFold(n_splits=cv)
    # shuffle the ID's (create a new column), and do splits based on new ID's
    random.seed(datetime.now())
    g = [df for _, df in df.groupby("ID")]
    random.shuffle(g)
    df = pd.concat(g).reset_index(drop=True)
    ids = df.groupby(["ID"], sort=False).ngroup()
    df["ID_shuffle"] = ids

    # for train_idx, test_idx in gkf.split(df, y=None, groups=df["ID_shuffle"]): #Updated for pandas new version
    for i, (train_idx, test_idx) in enumerate(gkf.split(df, y=None, groups=df["ID_shuffle"])):
        print(f"Training on Fold {i+1}")
        df_train = df[df.index.isin(train_idx)]
        df_test = df[df.index.isin(test_idx)]
        # Initialize Clusters
        df_init = initializeClusters(
            df_train,
            clustering=clustering,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            random_state=random_state,
        )
        # Run Iterative Learning mrl

        (
            df_new,
            incoherences,
            training_error,
            testing_error,
            best_df,
            opt_k,
            split_scores,
            _,
        ) = splitter(
            df_init,
            pfeatures,
            th,
            eta=eta,
            precision_thresh=precision_thresh,
            df_test=df_test,
            testing=True,
            max_k=max_k,
            classification=classification,
            split_classifier_params=split_classifier_params,
            h=h,
            gamma=gamma,
            verbose=False,
            n=n,
            plot=plot,
        )
        
        training_error.columns = ["Clusters", f"Error_{i}"]
        df_training_error = df_training_error.merge(
            training_error, how="outer", on=["Clusters"]
        )
        testing_error.columns = ["Clusters", f"Error_{i}"]
        df_testing_error = df_testing_error.merge(
            testing_error, how="outer", on=["Clusters"]
        )
        incoherences.columns = ["Clusters", f"Incoherence_{i}"]
        df_incoherences = df_incoherences.merge(
            incoherences, how="outer", on=["Clusters"]
        )

    df_training_error.set_index("Clusters", inplace=True)
    df_testing_error.set_index("Clusters", inplace=True)
    df_incoherences.set_index("Clusters", inplace=True)

    df_training_error.dropna(inplace=True)
    df_testing_error.dropna(inplace=True)
    df_incoherences.dropna(inplace=True)

    cv_training_error = np.mean(df_training_error, axis=1)
    cv_testing_error = np.mean(df_testing_error, axis=1)
    cv_incoherences = np.mean(df_incoherences, axis=1)

    if plot:
        fig1, ax1 = plt.subplots()
        ax1.plot(
            cv_training_error.index.values, cv_training_error, label="CV Training Error"
        )
        ax1.plot(
            cv_testing_error.index.values, cv_testing_error, label="CV Testing Error"
        )

        if n > 0:
            ax1.axvline(
                x=n, linestyle="--", color="r"
            )  # plotting vertical line at #cluster =n
        ax1.set_ylim(0)
        ax1.set_xlabel("# of Clusters")
        ax1.set_ylabel("Mean CV Error or Accuracy %")
        ax1.set_title("Mean CV Error and Accuracy During Splitting")
        ax1.legend()

    return (cv_incoherences, cv_training_error, cv_testing_error, split_scores)


def printmem(name):
    print(name)
    objmem = []
    for obj in gc.get_objects():
        if type(obj) in [pd.Series, pd.DataFrame]:
            objmem.append((obj, sys.getsizeof(obj)))
    tot_mem = 0
    for obj, mem in sorted(objmem, key=lambda x: x[1]):
        print(obj.name if hasattr(obj, "name") else "No Name", mem)
        tot_mem += mem
        if mem > 600000:
            print(obj.dtypes)
    print(sum([sys.getsizeof(obj) for obj in gc.get_objects()]))
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)  # in bytes
