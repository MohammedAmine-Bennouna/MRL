# -*- coding: utf-8 -*-
"""
This file is intended to perform various testing measurements on the output of 

the MDP Clustering mrl.

Created on Sun Apr 26 23:13:09 2020

@author: Amine
"""
#################################################################
# Load Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.series import Series
import graphviz
import math
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import time
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv

#################################################################


#################################################################
# Functions for Predictions

# predict_cluster() takes in a clustered dataframe df_new, the number of
# features pfeatures, and returns a prediction model m that predicts the most
# likely cluster from a datapoint's features
def predict_cluster(
    df_new, pfeatures  # dataframe: trained clusters
):  # int: # of features
    X = df_new.iloc[:, 2 : 2 + pfeatures]
    y = df_new["CLUSTER"]

    params = {"max_depth": [3, 4, 6, 10, None]}

    m = DecisionTreeClassifier()
    # m = RandomForestClassifier()

    m = GridSearchCV(m, params, cv=5)

    # m = DecisionTreeClassifier(max_depth = 10)
    m.fit(X, y)
    return m


# predict_value_of_cluster() takes in MDP parameters, a cluster label, and
# and a list of actions, and returns the predicted value of the given cluster
# currently takes value of current cluster as well as next cluster
def predict_value_of_cluster(
    P_df, R_df, cluster, actions  # df: MDP parameters  # int: cluster number
):  # list: list of actions
    s = cluster
    v = R_df.loc[s]
    for a in actions:
        s = P_df.loc[s, a].values[0]
        v = v + R_df.loc[s]
    return v


# function to generate a colormap for our plots with
# distinct colors
def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(
        math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades
    )

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = (
        np.arange(number_of_distinct_colors_with_multiply_of_shades)
        / number_of_distinct_colors_with_multiply_of_shades
    )

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(
        number_of_shades,
        number_of_distinct_colors_with_multiply_of_shades // number_of_shades,
    )

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8 / lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = (
                np.ones(number_of_shades)
                - initial_cm[
                    lower_half
                    + j * number_of_shades : lower_half
                    + (j + 1) * number_of_shades,
                    i,
                ]
            )
            modifier = j * modifier / upper_partitions_half
            initial_cm[
                lower_half
                + j * number_of_shades : lower_half
                + (j + 1) * number_of_shades,
                i,
            ] += modifier

    return ListedColormap(initial_cm)


# get_MDP() takes in a clustered dataframe df_new, and returns dataframes
# P_df and R_df that represent the parameters of the estimated MDP (if sink
# exists, it will be the last cluster and goes to itself)
def get_MDP(df_new):
    # print('df_new orig getMDP', df_new)
    # removing None values when counting where clusters go
    df0 = df_new[df_new["NEXT_CLUSTER"] != "None"]
    # print(df0[df0['CLUSTER']==3])
    # df0 = df_new
    transition_counts = df0.groupby(["CLUSTER", "ACTION", "NEXT_CLUSTER"]).size()

    # next cluster given how most datapionts transition for the given action
    transition_df = transition_counts.groupby(["CLUSTER", "ACTION"]).idxmax()

    P_df = pd.DataFrame()
    P_df["NEXT_CLUSTER"] = transition_df.apply(lambda x: x[2])
    R_df = df_new.groupby("CLUSTER")["RISK"].mean()

    # check if end state exists, if so make a sink node as the last cluster
    if "End" in list(P_df["NEXT_CLUSTER"].unique()):
        # print('Adding Sink Nodes...')
        P_df = P_df.reset_index()

        # find goal cluster that leads to sink, then remove
        # there are now multiple goal clusters so we have to fix this
        cs = P_df.loc[P_df["NEXT_CLUSTER"] == "End"]["CLUSTER"].unique()

        P_df_noend = P_df.loc[P_df["NEXT_CLUSTER"] != "End"]
        P_df, R_df = add_sink(P_df_noend, cs, R_df)

    return P_df, R_df


def get_MDP_stochastic(df_new):
    """
    Same as above. Stochastic.
    P_df now represented as c,a,c' -> p
    R_df is still c -> r
    Where c is current cluster, a is action take, c' next cluster, r reward, and p probability
    """
    df0 = df_new[df_new["NEXT_CLUSTER"] != "None"]

    s = df0["CLUSTER"].max() + 1
    df0.loc[df0["NEXT_CLUSTER"] == "End", "NEXT_CLUSTER"] == s
    actions = df0[df0["NEXT_CLUSTER"] != s]["ACTION"].unique()

    transition_counts = df0.groupby(["CLUSTER", "ACTION", "NEXT_CLUSTER"]).size()
    transition_df = (
        transition_counts / transition_counts.groupby(["CLUSTER", "ACTION"]).sum()
    )

    P_df = pd.DataFrame()
    P_df["PROBABILITY"] = transition_df
    P_df = P_df.reset_index()

    df_end = []
    for a in actions:
        df_end.append([s, a, s, 1])
    P_df = pd.concat(
    [P_df, pd.DataFrame(df_end, columns=["CLUSTER", "ACTION", "NEXT_CLUSTER", "PROBABILITY"])],
    ignore_index=True
    ) #DEBUG

    R_df = df_new.groupby("CLUSTER")["RISK"].mean()
    R_df = pd.concat([R_df, pd.Series([0], index=[s])], axis=1).T #DEBUG

    return P_df, R_df


def add_sink(P_df_noend, cs, R_df):
    """Add a sink node to the transition functions"""
    # create dataframe that goal go to sink and sink go to sink
    s = P_df_noend["CLUSTER"].max() + 1
    actions = P_df_noend["ACTION"].unique()
    df_end = []
    for a in actions:
        # s is sink/end state
        # success, failure (action == None)
        for c in cs:
            df_end.append([c, a, s])
        df_end.append([s, a, s])
    df_end = pd.DataFrame(df_end, columns=["CLUSTER", "ACTION", "NEXT_CLUSTER"])

    # P_df = P_df_noend.append(df_end) #Pandas Update
    P_df = pd.concat([P_df_noend, df_end], ignore_index=True)
    #-----

    P_df.sort_values(by=["CLUSTER", "ACTION"], inplace=True)
    P_df.set_index(["CLUSTER", "ACTION"], inplace=True)

    # set new reward node
    # R_df = R_df.append(pd.Series([0], index=[s])) #Pandas Update
    R_df = pd.concat([R_df, pd.Series([0], index=[s])])
    #-----
    return P_df, R_df


#################################################################


#################################################################
# Functions for Error

# training_value_error() takes in a clustered dataframe, and computes the
# E((\hat{v}-v)^2) expected error in estimating values (risk) given actions
# Returns a float of sqrt average value error per ID


def training_value_error(
    df_new,  # Outpul of algorithm
    gamma=1,  # discount factor
    relative=False,  # Output Raw error or RMSE ie ((\hat{v}-v)/v)^2
    h=5,  # Length of forecast. The error is computed on v_h = \sum_{t=h}^H v_t
    # if h = -1, we forecast the whole path
    eval_samples=None,  # Number of samples to evaluate. If None, evaluate on all
    num_sims=20,
    stochastic=False,
):  # Length of forecast. The error is computed on v_h = \sum_{t=h}^H v_t
    # if h = -1, we forecast the whole path
    E_v = 0
    P_df, R_df = get_MDP(df_new)
    df2 = df_new.reset_index()
    df2 = df2.groupby(["ID"]).first()
    N_train = df2.shape[0]

    # default evals are the N_train paths, but can specify sampling a certain number of paths
    eval_ids = list(range(N_train))
    if eval_samples:
        eval_ids = np.random.default_rng().choice(
            N_train, size=eval_samples, replace=False
        )

    for i in eval_ids:
        index = df2["index"].iloc[i]
        # initializing first state for each ID

        if h == -1:
            t = 0

        else:
            H = -1
            # Computing Horizon H of ID i
            while True:
                H += 1
                # tells us if this is the end of a path
                try:
                    df_new["ID"].loc[index + H + 1]
                except:
                    break
                if df_new["ID"].loc[index + H] != df_new["ID"].loc[index + H + 1]:
                    break
            t = H - h

        v_true = 0
        v_estims = []
        s = df_new["CLUSTER"].loc[index + t]
        a = df_new["ACTION"].loc[index + t]

        # only need 1 sim if we are not using stochastic
        if not stochastic:
            num_sims = 1

        # average error of num_sims
        for i in range(num_sims):

            v_estim = 0
            t = 0

            # predicting path of each ID
            while True:

                # only calculate v_true on the first iteration
                if i == 0:
                    v_true = gamma * v_true + df_new["RISK"].loc[index + t]
                v_estim = gamma * v_estim + R_df.loc[s]

                # this tells us if the path is over
                try:
                    df_new["ID"].loc[index + t + 1]
                except:
                    break
                if df_new["ID"].loc[index + t] != df_new["ID"].loc[index + t + 1]:
                    break

                try:
                    if not stochastic:
                        s = P_df.loc[s, a].values[0]
                    else:
                        s = sim_next_cluster(P_df, s, a)
                # error raises in case we never saw a given transition in the data
                # except ValueError
                except:
                    pass
                    # print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)

                t += 1
                a = df_new["ACTION"].loc[index + t]

            # add the calculations from this iteration
            if relative:
                E_v = E_v + ((v_true - v_estim) / v_true) ** 2
            else:
                E_v = E_v + (v_true - v_estim) ** 2

            v_estims.append(v_estim)

        v_estim = sum(v_estims) / len(v_estims)

    E_v = E_v / N_train
    return np.sqrt(E_v)


def sim_next_cluster(P_df, s, a):
    """Get the next cluster from cluster/action pair, stochastic case"""
    nc_probs = P_df[(P_df["CLUSTER"] == s) & (P_df["ACTION"] == a)]
    if not len(nc_probs):
        raise ValueError("Transition Action not observed")
    rand = np.random.random()

    agg_probs = 0
    for _, row in nc_probs.iterrows():
        agg_probs += row["PROBABILITY"]
        if agg_probs > rand:
            return row["NEXT_CLUSTER"]
    print(rand, agg_probs)
    raise ValueError("Probabilities dont sum to 1")


# testing_value_error() takes in a dataframe of testing data, and dataframe of
# new clustered data, a model from predict_cluster function, and computes the
# expected value error given actions and a predicted initial cluster and time
# horizon h (if h = -1, we forecast the whole path)
# Returns a float of sqrt average value error per ID
def testing_value_error(
    df_test, df_new, model, pfeatures, gamma=1, relative=False, h=5  # discount factor
):
    E_v = 0
    P_df, R_df = get_MDP(df_new)
    df2 = df_test.reset_index()
    df2 = df2.groupby(["ID"]).first()
    N_test = df2.shape[0]

    df_test = df_test.assign(CLUSTER=model.predict(df_test.iloc[:, 2 : 2 + pfeatures]))

    for i in range(N_test):
        # initializing index of first state for each ID
        index = df2["index"].iloc[i]
        cont = True

        if h == -1:
            t = 0
            t0 = 0

        else:
            H = -1
            # Computing Horizon H of ID i
            while cont:
                H += 1
                try:
                    df_test["ID"].loc[index + H + 1]
                except:
                    break
                if df_test["ID"].loc[index + H] != df_test["ID"].loc[index + H + 1]:
                    break
            t = H - h
            t0 = H - h

        v_true = 0
        v_estim = 0
        s = df_test["CLUSTER"].loc[index + t]
        a = df_test["ACTION"].loc[index + t]

        # predicting path of each ID
        while cont:
            v_true = v_true + df_test["RISK"].loc[index + t] * (gamma ** (t - t0))
            v_estim = v_estim + R_df.loc[s] * (gamma ** (t - t0))
            try:
                df_test["ID"].loc[index + t + 1]
            except:
                break
            if df_test["ID"].loc[index + t] != df_test["ID"].loc[index + t + 1]:
                break

            try:
                s = P_df.loc[s, a].values[0]
            # error raises in case we never saw a given transition in the data

            # except TypeError: # sometimes we see KeyError or IndexError...
            except:
                pass
                # print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)

            t += 1
            a = df_test["ACTION"].loc[index + t]
        if relative:
            E_v = E_v + ((v_true - v_estim) / v_true) ** 2
        else:
            E_v = E_v + (v_true - v_estim) ** 2

    E_v = E_v / N_test
    return np.sqrt(E_v)


#################################################################


#################################################################
# Functions for R2 Values

# R2_value_training() takes in a clustered dataframe, and returns a float
# of the R-squared value between the expected value and true value of samples
# currently doesn't support horizon h specifications
def R2_value_training(df_new, eval_samples=None):
    E_v = 0
    # print('dfnew none r2vt', df_new[df_new['ACTION'] == 'None'])
    print("Calculating MDP...")

    ts = time.time()
    P_df, R_df = get_MDP(df_new)
    print(f"MDP calculation took {time.time()-ts}")

    assert not P_df.index.duplicated().any(), f"Duplicate transitions: {P_df}"
    assert not R_df.index.duplicated().any(), f"Duplicate rewards: {R_df}"
    # print(P_df)
    df2 = df_new.reset_index()
    df2 = df2.groupby(["ID"]).first()
    N = df2.shape[0]
    V_true = []
    # print('df2 R2vt', df2)
    # print('P_df R2vt', P_df)
    print("Begin Main Calcuations for R2 metric...")

    eval_ids = list(range(N))
    if eval_samples:
        eval_ids = np.random.default_rng().choice(N, size=eval_samples, replace=False)

    for i in eval_ids:
        # initializing starting cluster and values
        s = df2["CLUSTER"].iloc[i]

        """  
        try:
            assert not hasattr(s, '__len__'), f'variable s ({s}, type {type(s)}) is list-like'
        except AssertionError:
            print(s, df2, i)
            raise
        """

        # print('s R2vt', s)
        a = df2["ACTION"].iloc[i]
        v_true = df2["RISK"].iloc[i]

        v_estim = R_df.loc[s]
        index = df2["index"].iloc[i]
        cont = True
        t = 1
        # iterating through path of ID
        while cont:
            v_true = v_true + df_new["RISK"].loc[index + t]
            try:
                df_new["ID"].loc[index + t + 1]
            except:
                break
            if df_new["ID"].loc[index + t] != df_new["ID"].loc[index + t + 1]:
                break

            # s_set = True
            # old_s = s
            try:
                next_s = P_df.loc[s, a]
                s = next_s.item()
            # error raises in case we never saw a given transition in the data
            # except TypeError:
            except KeyError:
                # s_set = False
                pass
                # print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_new["ACTION"].loc[index + t]
            # print('R_df R2vt', R_df, 's', s)
            """
            try:
                assert not hasattr(s, '__len__'), f'variable s ({s}, type {type(s)}) is list-like'
            except AssertionError:
                print('start debug printing')
                print(s) 
                print(P_df) 
                print(a) 
                print(s_set) 
                print(old_s)
                print(P_df.loc[old_s, a].values)
                print('end debug printing')
                raise
            v_estim = v_estim + R_df.loc[s]
            #print('vestim R2vt', v_estim)
            """
            t += 1
        # assert type(v_true) not in [pd.DataFrame, pd.Series], f'bad type vtrue {type(v_true)}'
        # assert type(v_estim) not in [pd.DataFrame, pd.Series], f'bad type vestim {type(v_estim)}'
        E_v = E_v + (v_true - v_estim) ** 2
        V_true.append(v_true)
    # defining R2 baseline & calculating the value
    E_v = E_v / N
    # print('E_v R2vt', E_v)
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    SS_tot = sum((V_true - v_mean) ** 2) / N
    return max(1 - E_v / (SS_tot + 1e-6 * E_v), 0)


# R2_value_testing() takes a dataframe of testing data, a clustered dataframe,
# a model outputted by predict_cluster, and returns a float of the R-squared
# value between the expected value and true value of samples in the test set
# currently doesn't support horizon h specifications
def R2_value_testing(df_test, df_new, model, pfeatures):
    E_v = 0
    P_df, R_df = get_MDP(df_new)
    df2 = df_test.reset_index()
    df2 = df2.groupby(["ID"]).first()
    N = df2.shape[0]

    # predicting clusters based on features
    clusters = model.predict(df2.iloc[:, 2 : 2 + pfeatures])
    df2["CLUSTER"] = clusters

    V_true = []
    for i in range(N):
        s = df2["CLUSTER"].iloc[i]
        a = df2["ACTION"].iloc[i]
        v_true = df2["RISK"].iloc[i]

        v_estim = R_df.loc[s]
        index = df2["index"].iloc[i]
        cont = True
        t = 1
        while cont:
            v_true = v_true + df_test["RISK"].loc[index + t]

            try:
                s = P_df.loc[s, a].values[0]
            # error raises in case we never saw a given transition in the data
            # except TypeError:
            except:
                pass
                # print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_test["ACTION"].loc[index + t]

            v_estim = v_estim + R_df.loc[s]
            try:
                df_test["ID"].loc[index + t + 1]
            except:
                break
            if df_test["ID"].loc[index + t] != df_test["ID"].loc[index + t + 1]:
                break
            t += 1
        E_v = E_v + (v_true - v_estim) ** 2
        V_true.append(v_true)
    E_v = E_v / N
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    SS_tot = sum((V_true - v_mean) ** 2) / N
    return max(1 - E_v / SS_tot, 0)


#################################################################


#################################################################
# Functions for Plotting and Visualization

# plot_features() takes in a dataframe and two features, and plots the data
# to illustrate the noise in each cluster
def plot_features(df, x, y, c="CLUSTER", ax=None, title=None, cmap="tab20", size=5):
    if ax == None:
        fig, ax = plt.subplots()

    df.plot(kind="scatter", x=x, y=y, c=c, cmap=cmap, ax=ax, s=3)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    # xticks = [0, 1, 2, 3, 4, 5]
    xticks = [k for k in range(size+1)]
    plt.xticks(xticks)
    yticks = [0, -1, -2, -3, -4, -5]
    yticks = [-k for k in range(size+1)]
    plt.yticks(yticks)
    if title != None:
        plt.title(title)
    # ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    # plt.axis('scaled')
    # plt.show()


# cluster_size() takes a dataframe, and returns the main statistics of each
# cluster in a dataframe
def cluster_size(df):
    df2 = df.groupby("CLUSTER")["RISK"].agg(["count", "mean", "std", "min", "max"])
    df2["rel"] = 100 * abs(df2["std"] / df2["mean"])
    df2["rel_mean"] = 100 * abs(df2["std"] / df["RISK"].mean())
    return df2


def next_clusters(df):
    """
    next_clusters() takes a dataframe, and returns a chart showing transitions from
    each cluster/action pair, count of each cluster/action pair, and the purity 
    of the highest next_cluster. 
    Disregards those with 'NEXT_CLUSTER' = None, and returns a dataframe of the results

    parameters
    ----------
    df : dataframe
        this is data in usual records format, the features, actions, rewards predicted clusters, 
            and predicted next clusters are some of the columns
    returns
    -------
    a dataframe with the following columns:
        cluster (c), action (a), count(c, a), purity(c, a)
        count(c, a) counts the number of times this cluster-action pair has occured in the data
        purity(c, a) = max_c' (count(c, a, c')/count(c, a)) is the proportion of data points in this cluster-action pair 
            which went to the most common next cluster
    """
    df = df.loc[df["NEXT_CLUSTER"] != "None"]
    df2 = df.groupby(["CLUSTER", "ACTION", "NEXT_CLUSTER"])["RISK"].agg(["count"])
    df2["purity"] = df2["count"] / df.groupby(["CLUSTER", "ACTION"])["RISK"].count()
    df2.reset_index(inplace=True)
    idx = df2.groupby(["CLUSTER", "ACTION"])["count"].transform(max) == df2["count"]
    df_final = df2[idx].groupby(["CLUSTER", "ACTION"]).max()
    df_final["count"] = df2.groupby(["CLUSTER", "ACTION"])["count"].sum()
    return df_final


def next_cluster_predictability(df):  # , regressor):  TODO: seems incomplete, ask about
    """
    Generates the necessary statistics for predicting the next cluster
    1. Counts the number of observations taking a given (clus, action) pair
    2. Fits a classification model to predict the next cluster based on the features,
    the loss function (e.g. Hinge Loss, for Lipschitz conditions) 
    """
    df = df.loc[df["NEXT_CLUSTER"] != "None"]
    df_final = df.groupby(["CLUSTER", "ACTION"])["RISK"].count()
    # df_final.reset_index(inplace=True) TODO: ask about bug
    return df_final


# decision_tree_diagram() takes in a trained MDP model, outputs a pdf of
# the best decision tree, as well as other visualizations
def decision_tree_diagram(model):
    # assumes that m.m, the prediction model, is a GridSearchCV object
    dc = model.m.best_estimator_

    # creating the decision tree diagram in pdf:
    dot_data = tree.export_graphviz(
        dc, out_file=None, filled=True, rounded=True, special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render("Decision_Tree_Diagram")

    return graph


# decision_tree_regions() takes a model and plots a visualization of the
# decision regions of two of the features (currently first and second)
def decision_tree_regions(model):
    dc = model.m.best_estimator_
    n_classes = model.df_trained["CLUSTER"].max()
    plot_step = 0.02

    plt.subplot()
    x_min = model.df_trained.iloc[:, 2].min() - 1
    x_max = model.df_trained.iloc[:, 2].max() + 1
    y_min = model.df_trained.iloc[:, 3].min() - 1
    y_max = model.df_trained.iloc[:, 3].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )

    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = dc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    for i in range(n_classes):
        idx = np.where(model.df_trained["CLUSTER"] == i)

        r = random.random()
        b = random.random()
        g = random.random()
        color = np.array([[r, g, b]])
        # colors = ['r', 'y', 'b']
        # color = colors[i%3]
        plt.scatter(
            model.df_trained.iloc[idx].iloc[:, 2],
            model.df_trained.iloc[idx].iloc[:, 3],
            c=color,
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

    plt.show()
    return


# model_trajectory() takes a trained model, the real transition function of
# the model f(x, u), the initial state x, and plots how the model's optimal
# policy looks like on the start state according to f1 and f2 two features
# indices e.g. x[f1] x[f2] plotted on the x and y axes, for n steps
def model_trajectory(
    m, f, x, f1=0, f2=None, n=50,size=5  # if f2 is none, only plot f1 over time
):
    states = []
    all_vecs = []
    if m.v is None:
        m.solve_MDP()

    if f2 != None:
        ys = [x[f2]]
        xs = [x[f1]]
    else:
        ys = [x[f1]]
        xs = range(n + 1)

    for i in range(n):
        # find current state and action
        s = m.m.predict(np.array(x).reshape(1, -1))
        # print(s)
        a = int(m.pi[s])
        # print(a)
        states.append([s, a])
        x_new = f(x, a)
        if x_new[0] == None:
            break

        if f2 != None:
            ys.append(x_new[f2])
            xs.append(x_new[f1])
        else:
            ys.append(x_new[f1])
        all_vecs.append(x_new)
        x = x_new
    # print('states', states, flush=True)
    # TODO: not plot the sink
    xs = np.array(xs)
    ys = np.array(ys)

    u = np.diff(xs)
    v = np.diff(ys)
    pos_x = xs[:-1] + u / 2
    pos_y = ys[:-1] + v / 2
    norm = np.sqrt(u ** 2 + v ** 2)

    fig, ax = plt.subplots()
    ax.set_xticks([k for k in range(size+1)])
    ax.set_yticks([-k for k in range(size+1)])
    ax.set_xlim(0 - 0.5, size + 0.5)
    ax.set_ylim(-size - 0.5, 0 + 0.5)
    ax.plot(xs, ys, marker="o")
    ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
    # ax.set_xlabel('FEATURE_%i' %f1)
    # ax.set_ylabel('FEATURE_%i' %f2)

    # set plot limits if relevant
    # plt.ylim(-l+0.5, 0.5)
    # plt.xlim(-.5, l-0.5)
    plt.show()
    return xs, ys, all_vecs


# plot_CV_training() takes a model trained by cross validation, and plots the
# testing error, training, error, and incoherence on the same graph
def plot_CV_training(model):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("K meta-state space size")
    ax1.set_ylabel("Score")
    ax1.plot(model.CV_error_all["Training Error"], label="In-Sample")
    ax1.plot(model.CV_error_all["Testing Error"], label="Testing")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:red"
    ax2.set_ylabel("Number of Incoherences")  # we already handled the x-label with ax1
    ax2.plot(model.CV_error_all["Incoherence"], color=color, label="Incoherences")

    fig.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


#################################################################


#################################################################
# Functions for Grid Testing (Predictions, Accuracy, Purity)

# get_predictions() takes in a clustered dataframe df_new, and maps each
# CLUSTER to an OG_CLUSTER that has the most elements
# Returns a dataframe of the mappings
def get_predictions(df_new):
    df0 = df_new.groupby(["CLUSTER", "OG_CLUSTER"])["ACTION"].count()
    df0 = df0.groupby("CLUSTER").idxmax()
    df2 = pd.DataFrame()
    df2["OG_CLUSTER"] = df0.apply(lambda x: x[1])
    return df2


# training_accuracy() takes in a clustered dataframe df_new, and returns the
# average training accuracy of all clusters (float) and a dataframe of
# training accuracies for each OG_CLUSTER
def training_accuracy(df_new):
    clusters = get_predictions(df_new)
    #    print('Clusters', clusters)

    # Tallies datapoints where the algorithm correctly classified a datapoint's
    # original cluster to be the OG_CLUSTER mapping of its current cluster
    accuracy = (
        clusters.loc[df_new["CLUSTER"]].reset_index()["OG_CLUSTER"]
        == df_new.reset_index()["OG_CLUSTER"]
    )
    # print(accuracy)
    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame("Accuracy")
    accuracy_df["OG_CLUSTER"] = df_new.reset_index()["OG_CLUSTER"]
    accuracy_df = accuracy_df.groupby("OG_CLUSTER").mean()
    return (tr_accuracy, accuracy_df)


# testing_accuracy() takes in a testing dataframe df_test (unclustered),
# a df_new clustered dataset, a model from predict_cluster and
# Returns a float for the testing accuracy measuring how well the model places
# testing data into the right cluster (mapped from OG_CLUSTER), and
# also returns a dataframe that has testing accuracies for each OG_CLUSTER
def testing_accuracy(
    df_test,  # dataframe: testing data
    df_new,  # dataframe: clustered on training data
    model,  # function: output of predict_cluster, mdp.m
    pfeatures,
):  # int: # of features

    clusters = get_predictions(df_new)

    test_clusters = model.predict(df_test.iloc[:, 2 : 2 + pfeatures])
    df_test = df_test.assign(CLUSTER=test_clusters)

    for cluster in df_test["CLUSTER"].unique():
        if cluster not in clusters.index:
            clusters.loc[cluster] = None

    # clusters = clusters.reset_index()
    # df_test = df_test[df_test['CLUSTER'] in clusters]

    accuracy = (
        clusters.loc[df_test["CLUSTER"]].reset_index()["OG_CLUSTER"]
        == df_test.reset_index()["OG_CLUSTER"]
    )

    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame("Accuracy")
    accuracy_df["OG_CLUSTER"] = df_test.reset_index()["OG_CLUSTER"]
    accuracy_df = accuracy_df.groupby("OG_CLUSTER").mean()
    return (tr_accuracy, accuracy_df)


# purity() takes a clustered dataframe and returns a dataframe with the purity
# of each cluster
def purity(df):
    su = pd.DataFrame(
        df.groupby(["CLUSTER"])["OG_CLUSTER"].value_counts(normalize=True)
    ).reset_index(level=0)
    su.columns = ["CLUSTER", "Purity"]
    return su.groupby("CLUSTER")["Purity"].max()


# generalization_accuracy() plots the training and testing accuracies as above
# for a given list of models and a test-set.
def generalization_accuracy(models, df_test, Ns):
    tr_accs = []
    test_accs = []
    for model in models:
        tr_acc, df = training_accuracy(model.df_trained)
        tr_accs.append(tr_acc)

        test_acc, df_t = testing_accuracy(
            df_test, model.df_trained, model.m, model.pfeatures
        )
        test_accs.append(test_acc)

    fig1, ax1 = plt.subplots()
    ax1.plot(Ns, tr_accs, label="Training Accuracy")
    ax1.plot(Ns, test_accs, label="Testing Accuracy")
    ax1.set_xlabel("N training data size")
    ax1.set_ylabel("Accuracy %")
    ax1.set_title("Model Generalization Accuracies")
    plt.legend()
    plt.show()
    return tr_accs, test_accs


# policy_accuracy() takes a trained model, and a df of optimal policies,
# and iterates through each row of the df to compare the model's predicted
# action with that of the real action taken in the data
# returns a percentage accuracy
def policy_accuracy(m, df):
    if m.v is None:
        m.solve_MDP()

    correct = 0
    df = df.loc[df["ACTION"] != "None"]
    # iterating through every line and comparing
    for index, row in df.iterrows():
        # predicted action:
        s = m.m.predict(np.array(row[2 : 2 + m.pfeatures]).reshape(1, -1))
        # s = m.df_trained.iloc[index]['CLUSTER']
        a = m.pi[s]

        # real action:
        a_true = row["ACTION"]
        if a == a_true:
            correct += 1
    total = df.shape[0]
    return correct / total


def policy_performance(policy, env, is_goal, r_goal, gamma=1, n_tests=20, T_max=1000):
    """
    Performance of a policy on an environment
    policy: a policy with get_action function
    env, is_goal, r_goal: specifies the environment with goal modifications
    n_tests: number of times to test
    T_max: max iteration for each test"""
    sum_rewards = 0
    for _ in range(n_tests):
        obs = env.reset()
        total_reward = 0
        for t in range(T_max):
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward * (gamma ** t)
            if done:
                if is_goal(obs):
                    total_reward += r_goal * (gamma ** t)
                break
        sum_rewards += total_reward
    return sum_rewards / n_tests


#################################################################
