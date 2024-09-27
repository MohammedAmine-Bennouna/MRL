#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Load Libraries
import pandas as pd
import numpy as np

from clustering import fit_CV, initializeClusters, splitter
from testing import (
    predict_cluster,
    training_value_error,
    get_MDP,
    predict_value_of_cluster,
    testing_value_error,
    model_trajectory,
    next_clusters,
    get_MDP_stochastic,
    next_cluster_predictability,
)
from mdp_utils import SolveMDP
from sklearn.metrics import accuracy_score
from scipy.stats import binom


class MDP_model:
    def __init__(self):
        self.df = None  # original dataframe from data
        self.pfeatures = None  # number of features
        self.CV_error = None  # error at minimum point of CV
        self.CV_error_all = None  # errors of different clusters after CV
        self.training_error = None  # training errors after last split sequence
        self.split_scores = None  # cv error from splitter (if GridSearch used)
        self.opt_k = None  # number of clusters in optimal clustering
        self.eta = None  # incoherence threshold
        self.df_trained = None  # dataframe after optimal training
        self.m = None  # model for predicting cluster number from features #CHANGE NAME
        self.clus_pred_accuracy = (
            None  # accuracy score of the cluster prediction function
        )
        self.P_df = None  # Transition function of the learnt MDP, includes sink node if end state exists
        self.R_df = None  # Reward function of the learnt MDP, includes sink node of reward 0 if end state exists
        self.nc = None  # dataframe similar to P_df, but also includes 'count' and 'purity' cols
        self.v = None  # value after MDP solved
        self.pi = None  # policy after MDP solved
        self.P = None  # P_df but in matrix form of P[a, s, s'], with alterations
        # where transitions that do not pass the action and purity thresholds
        # now lead to a new cluster with high negative reward
        self.R = None  # R_df but in matrix form of R[a, s]

    def load_df(self, df_clustered, pfeatures, opt_k=None):
        """"
        Loads a MRL model from the clustered dataframe. Calculates other values as in fit.

        Args:
            df_clustered (pd.DataFrame): Clustered dataframe.
            pfeatures (int): Number of features.
            opt_k (int, optional): Optimal number of clusters. Defaults to None.
        """
        self.df = df_clustered.copy()
        self.pfeatures = pfeatures
        self.df_trained = df_clustered.copy()
        self.opt_k = opt_k

        self.create_model()

    def get_info(self):
        "Print an extensive summary of the model."
        #DEBUG #To code
        return None

    # fit_CV() takes in parameters for prediction, and trains the model on the
    # optimal clustering for a given horizon h (# of actions), using cross
    # validation. See fit_CV in clustering.py for further documentation.
    def fit_CV(
        self,
        data,  # needs a dataframe where 'ACTION' == 'None' if goal state is reached.
        pfeatures,
        h=5,
        gamma=1,
        max_k=70,
        distance_threshold=0.05,
        cv=5,
        th=0,
        eta=float("inf"),  # calculated by eta*sqrt(datapoints)/clusters
        precision_thresh=1e-14,
        classification="DecisionTreeClassifier",
        split_classifier_params={"random_state": 0},
        clustering="Agglomerative",
        n_clusters=None,
        random_state=0,
        plot=False,
        verbose=False,
    ):
        """
        Trains the model on the optimal clustering for a given horizon using cross-validation.

        Args:
            data (pd.DataFrame): Input dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION'].
            pfeatures (int): Number of features.
            h (int, optional): Time horizon. Defaults to 5.
            gamma (float, optional): Discount value. Defaults to 1.
            max_k (int, optional): Maximum number of clusters. Defaults to 70.
            distance_threshold (float, optional): Clustering diameter for Agglomerative clustering. Defaults to 0.05.
            cv (int, optional): Number of folds for cross-validation. Defaults to 5.
            th (int, optional): Splitting threshold. Defaults to 0.
            eta (float, optional): Incoherence threshold. Defaults to float('inf').
            precision_thresh (float, optional): Precision threshold. Defaults to 1e-14.
            classification (str, optional): Classification method. Defaults to 'DecisionTreeClassifier'.
            split_classifier_params (dict, optional): Parameters for the classification method. Defaults to {'random_state': 0}.
            clustering (str, optional): Clustering method. Defaults to 'Agglomerative'.
            n_clusters (int, optional): Number of clusters for KMeans. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 0.
            plot (bool, optional): Flag to plot the results. Defaults to False.
            verbose (bool, optional): Verbosity flag. Defaults to False.
        """

        df = data.copy()

        # save relevant data
        self.df = df
        self.pfeatures = pfeatures
        self.eta = eta

        # run cross validation on the data to find best clusters

        cv_incoherences, cv_training_error, cv_testing_error, split_scores = fit_CV(
            self.df,
            self.pfeatures,
            th=th,
            clustering=clustering,
            distance_threshold=distance_threshold,
            eta=eta,
            precision_thresh=precision_thresh,
            classification=classification,
            split_classifier_params=split_classifier_params,
            max_k=max_k,
            n_clusters=n_clusters,
            random_state=random_state,
            h=h,
            gamma=gamma,
            verbose=verbose,
            cv=cv,
            n=-1,
            plot=plot,
        )

        # store cv testing error
        cv_testing_error = pd.concat(
            [
                cv_testing_error.rename("Testing Error"),
                cv_training_error.rename("Training Error"),
                cv_incoherences.rename("Incoherence"),
            ],
            axis=1,
        )
        self.CV_error_all = cv_testing_error

        cv_testing_error.reset_index(inplace=True)

        # find optimal cluster after filtering for eta
        inc_thresh = self.eta * self.df.shape[0] ** 0.5
        filtered = cv_testing_error.loc[
            cv_testing_error["Incoherence"]
            < inc_thresh / (cv_testing_error["Clusters"])
        ]

        filtered.set_index("Clusters", inplace=True)
        k = filtered["Testing Error"].idxmin()

        if verbose:
            print("CV Testing Error")
            print(cv_testing_error)
            print("best clusters:", k)
        self.opt_k = k

        # actual training on all the data
        df_init = initializeClusters(
            self.df,
            clustering=clustering,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            random_state=random_state,
        )

        # Rename end state to 'end'
        df_init.loc[df_init["ACTION"] == "None", "NEXT_CLUSTER"] = "End"

        (
            df_new,
            df_incoherences,
            training_error,
            testing_error,
            best_df,
            opt_k,
            split_scores,
            stoc_training_error,
        ) = splitter(
            df_init,
            pfeatures=self.pfeatures,
            th=th,
            eta=self.eta,
            precision_thresh=precision_thresh,
            df_test=None,
            testing=False,
            max_k=self.opt_k,
            classification=classification,
            split_classifier_params=split_classifier_params,
            h=h,
            gamma=gamma,
            verbose=verbose,
            plot=plot,
        )

        # storing trained dataset and predict_cluster function and accuracy
        self.df_trained = df_new
        self.m = predict_cluster(df_new, self.pfeatures)
        pred = self.m.predict(df_new.iloc[:, 2 : 2 + self.pfeatures])
        self.clus_pred_accuracy = accuracy_score(pred, df_new["CLUSTER"])

        # store final training error and incoherenes
        self.training_error = training_value_error(self.df_trained)
        self.incoherences = df_incoherences
        self.split_scores = split_scores

        # store P_df and R_df values
        P_df, R_df = get_MDP(self.df_trained)
        self.P_df = P_df
        self.R_df = R_df

        # store statistics and clusters transitions
        self.nc = next_clusters(df_new)

    # fit() takes in the parameters for prediction, and directly fits the model
    # to the data without running cross validation. If optimize is set to True,
    # stores the best clustering in self.df_trained; otherwise stores the
    # clustering at when max_k number of clusters is reached.
    def fit(
        self,
        data,  # needs a dataframe where 'ACTION' == 'None' if goal state is reached.
        pfeatures,
        h=5,
        gamma=1,
        max_k=70,
        distance_threshold=0.05,
        cv=5,
        th=0,
        eta=float("inf"),
        precision_thresh=1e-14,
        classification="DecisionTreeClassifier",
        split_classifier_params={"random_state": 0, "min_impurity_decrease": 0.02},
        clustering="Agglomerative",
        n_clusters=None,
        random_state=0,
        plot=False,
        optimize=True,
        verbose=False,
        save_epoch=False,
        save_path=None,
        save_every=1,
        eval_samples=None,
        stochastic=False,
    ):

        """
        Directly fits the model to the data without running cross-validation.

        Args:
            data (pd.DataFrame): Input dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION'].
            pfeatures (int): Number of features.
            h (int, optional): Time horizon. Defaults to 5.
            gamma (float, optional): Discount value. Defaults to 1.
            max_k (int, optional): Maximum number of clusters. Defaults to 70.
            distance_threshold (float, optional): Clustering diameter for Agglomerative clustering. Defaults to 0.05.
            cv (int, optional): Number for cross-validation. Defaults to 5.
            th (int, optional): Splitting threshold. Defaults to 0.
            eta (float, optional): Incoherence threshold. Defaults to float('inf').
            precision_thresh (float, optional): Precision threshold. Defaults to 1e-14.
            classification (str, optional): Classification method. Defaults to 'DecisionTreeClassifier'.
            split_classifier_params (dict, optional): Parameters for the classification method. Defaults to {'random_state': 0, 'min_impurity_decrease': 0.02}.
            clustering (str, optional): Clustering method. Defaults to 'Agglomerative'.
            n_clusters (int, optional): Number of clusters for KMeans. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 0.
            plot (bool, optional): Flag to plot the results. Defaults to False.
            optimize (bool, optional): Flag to optimize the clusters. Defaults to True.
            verbose (bool, optional): Verbosity flag. Defaults to False.
            save_epoch (bool, optional): Flag to save the model at each epoch. Defaults to False.
            save_path (str, optional): Path to save the model. Defaults to None.
            save_every (int, optional): Interval to save the model. Defaults to 1.
            eval_samples (pd.DataFrame, optional): Evaluation samples. Defaults to None.
            stochastic (bool, optional): Flag to use stochastic MDP. Defaults to False.
        """

        df = data.copy()
        print("copied data")

        # save relevant data
        self.df = df
        self.pfeatures = pfeatures
        self.eta = eta
        self.t_max = df["TIME"].max()
        self.r_max = abs(df["RISK"]).max()

        print("initializing clusters...")

        # training on all the data
        df_init = initializeClusters(
            self.df,
            clustering=clustering,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            random_state=random_state,
        )

        # print('df init model.fit', df_init)

        print("Clusters Initialized")
        if verbose:
            print(df_init)

        if save_epoch:
            print("Saving initial df")
            save_path.mkdir(exist_ok=True)
            df_init.to_csv(save_path / f"df_epoch_-1.csv")

        (
            df_new,
            df_incoherences,
            training_error,
            testing_error,
            best_df,
            opt_k,
            split_scores,
            stoc_training_error,
        ) = splitter(
            df_init,
            pfeatures=self.pfeatures,
            th=th,
            eta=self.eta,
            precision_thresh=precision_thresh,
            df_test=None,
            testing=False,
            max_k=max_k,
            classification=classification,
            split_classifier_params=split_classifier_params,
            h=h,
            gamma=gamma,
            verbose=verbose,
            plot=plot,
            save_epoch=save_epoch,
            save_path=save_path,
            save_every=save_every,
            eval_samples=eval_samples,
            stochastic=stochastic,
        )

        # store all training errors
        self.training_error = training_error
        self.incoherences = df_incoherences
        self.split_scores = split_scores

        # storing trained dataset and predict_cluster function, depending on
        # whether optimization was selected
        # incoherence and precision thresholds were already applied
        # within splitter to find best_df and opt_k
        if optimize:
            self.df_trained = best_df
            # k = self.training_error['Clusters'].iloc[self.training_error['Error'].idxmin()]
            self.opt_k = opt_k
        else:
            self.df_trained = df_new
            self.opt_k = self.training_error["Clusters"].max()

        self.create_model(stochastic=stochastic)

    def create_model(self, stochastic):
        """
        After MRL is trained. Creates the underlying MDP model by fitting a decision tree 
        to predict clusters, then calculating the empirical transition functions."""
        self.m = predict_cluster(self.df_trained, self.pfeatures)
        pred = self.m.predict(self.df_trained.iloc[:, 2 : 2 + self.pfeatures])
        self.clus_pred_accuracy = accuracy_score(pred, self.df_trained["CLUSTER"])

        if not stochastic:
            # store P_df and R_df values
            P_df, R_df = get_MDP(self.df_trained)
            self.P_df = P_df
            self.R_df = R_df

            # store next_clusters dataframe
            self.nc = next_clusters(self.df_trained)

        if stochastic:
            P_df, R_df = get_MDP_stochastic(self.df_trained)
            self.P_df = P_df
            self.R_df = R_df

            # nc_predictability is used for robustness
            self.nc_predictability = next_cluster_predictability(self.df_trained,)

    # predict() takes a list of features and a time horizon, and returns
    # the predicted value after all actions are taken in order
    def predict(
        self, features, actions  # list: list OR array of features
    ):  # list: list of actions

        # predict initial cluster
        s = int(self.m.predict([features]))

        # predict value sum given starting cluster and action path
        v = predict_value_of_cluster(self.P_df, self.R_df, s, actions)
        return v

    # predict_forward() takes an ID & actions, and returns the predicted value
    # for this ID after all actions are taken in order
    def predict_forward(self, ID, actions):

        # cluster of this last point
        s = self.df_trained[self.df_trained["ID"] == ID].iloc[-1, -2]

        # predict value sum given starting cluster and action path
        v = predict_value_of_cluster(self.P_df, self.R_df, s, actions)
        return v

    # testing_error() takes a df_test, then computes and returns the testing
    # error on this trained model
    def testing_error(self, df_test, relative=False, h=-1):

        error = testing_value_error(
            df_test, self.df_trained, self.m, self.pfeatures, relative=relative, h=h
        )

        return error

    # solve_MDP() takes the trained model as well as parameters for gamma,
    # epsilon, whether the problem is a minimization or maximization one,
    # and the threshold cutoffs to not include actions that don't appear enough
    # in each state, as well as purity cutoff for next_states that do not
    # represent enough percentage of all the potential next_states,
    # and returns the the value and policy. When solving the MDP, creates an
    # artificial punishment state that is reached for state/action pairs that
    # don't meet the above cutoffs; also creates a sink node of reward 0
    # after the goal state or punishment state is reached.
    def solve_MDP(
        self,
        alpha=0.2,  # Robustness param (0-1): 0 is most conservative (see comment for details)
        beta=0.8,  # Robustness param (0-1): 1 is most conservative (see comment for details)
        min_action_obs=-1,  # least number of actions that must be seen
        min_action_purity=0.7,  # float: percentage purity above which is acceptable
        prob="max",  # str: 'max', or 'min' for maximization or minimization problem
        gamma=1,  # discount factor
        epsilon=10 ** (-8),
        p=False,
    ):
        """
        When constricting an MDP from leanrt partition (clustering), 
        the following tests are used for robustness in solve_MDP. For every transition (cluster, action, next_cluster),
        we discard the transition from the learnt MDP if it doesn't pass the following tests:
        1. Binomial CDF statistics test:
        We assume null hypothesis that <=beta of our data is actually going into the correct cluster, and seek to 
            reject this hypothesis with significance p<=alpha. If we fail to reject, then we remove this cluster-action pair.
        2. Min Observation: We check if we observed the transition sufficently enough
        3. Min purity: We check whether the vast majority of data points indeed had the same transition with the given action. 
            We compare the ratio to a treshold.
        """
        self.create_PR(alpha, beta, min_action_obs, min_action_purity, prob)
        return self.solve_helper(gamma, epsilon, p, prob)

#------------------- Deterministic MRL -------------------#
    def create_PR(self, alpha, beta, min_action_obs, min_action_purity, prob):

        """
        When constricting an MDP from leanrt partition (clustering), 
        the following tests are used for robustness in solve_MDP. For every transition (cluster, action, next_cluster),
        we discard the transition from the learnt MDP if it doesn't pass the following tests:
        1. Binomial CDF statistics test:
        We assume null hypothesis that <=beta of our data is actually going into the correct cluster, and seek to 
            reject this hypothesis with significance p<=alpha. If we fail to reject, then we remove this cluster-action pair.
        2. Min Observation: We check if we observed the transition sufficently enough
        3. Min purity: We check whether the vast majority of data points indeed had the same transition with the given action. 
            We compare the ratio to a treshold.
        """
        
        # if default value, then scale the min threshold with data size, ratio 0.008
        if min_action_obs == -1:
            min_action_obs = max(5, 0.008 * self.df_trained.shape[0])

        # adding two clusters: one for sink node (reward = 0), one for punishment state
        # sink node is R[s-2], punishment state is R[s-1]

        P_df = self.P_df.copy()
        R_df = self.R_df.copy()
        P_df["count"] = self.nc["count"]
        P_df["purity"] = self.nc["purity"]
        P_df = P_df.reset_index()
        R_df = R_df.reset_index()

        # record parameters of transition dataframe
        a = P_df["ACTION"].nunique()
        s = P_df["CLUSTER"].nunique()
        actions = P_df["ACTION"].unique()

        # Take out rows that don't pass statistical alpha test
        P_alph = P_df.loc[
            (1 - binom.cdf(P_df["purity"] * (P_df["count"]), P_df["count"], beta))
            <= alpha
        ]

        # Take out rows where actions or purity below threshold
        P_thresh = P_alph.loc[
            (P_alph["count"] > min_action_obs) & (P_alph["purity"] > min_action_purity)
        ]

        # Take note of rows where we have missing actions:
        incomplete_clusters = np.where(P_df.groupby("CLUSTER")["ACTION"].count() < a)[0]
        missing_pairs = []
        for c in incomplete_clusters:
            not_present = np.setdiff1d(
                actions, P_df.loc[P_df["CLUSTER"] == c]["ACTION"].unique()
            )
            for u in not_present:
                missing_pairs.append((c, u))

        P = np.zeros((a, s + 1, s + 1))

        # model transitions
        for row in P_thresh.itertuples():
            x, y, z = row[2], row[1], row[3]  # ACTION, CLUSTER, NEXT_CLUSTER
            P[x, y, z] = 1

        # reinsert transition for cluster/action pairs taken out by alpha test
        excl_alph = P_df.loc[
            (1 - binom.cdf(P_df["purity"] * P_df["count"], P_df["count"], beta)) > alpha
        ]
        for row in excl_alph.itertuples():
            c, u = row[1], row[2]  # CLUSTER, ACTION
            P[u, c, -1] = 1

        # reinsert transition for cluster/action pairs taken out by threshold
        excl = P_df.loc[
            (P_df["count"] <= min_action_obs) | (P_df["purity"] <= min_action_purity)
        ]
        for row in excl.itertuples():
            c, u = row[1], row[2]  # CLUSTER, ACTION
            P[u, c, -1] = 1

        # reinsert transition for missing cluster-action pairs
        for pair in missing_pairs:
            c, u = pair
            P[u, c, -1] = 1

        # replacing correct sink node transitions
        nan = P_df.loc[P_df["count"].isnull()]
        for row in nan.itertuples():
            c, u, t = row[1], row[2], row[3]  # CLUSTER, ACTION, NEXT_CLUSTER
            P[u, c, t] = 1

        # punishment node to 0 reward sink (if sink was created in get_MDP):
        if "End" in self.df_trained["NEXT_CLUSTER"].unique():
            for u in range(a):
                P[u, -1, -2] = 1

        # append high negative reward for incorrect / impure transitions
        R = []

        T_max = self.df_trained["TIME"].max()
        r_max = abs(self.df_trained["RISK"]).max()
        self.t_max = T_max
        self.r_max = r_max
        for i in range(a):
            if prob == "max":
                # take T-max * max(abs(reward)) * 2
                R.append(np.append(np.array(self.R_df), -self.t_max * self.r_max * 2))
            else:
                R.append(np.append(np.array(self.R_df), self.t_max * self.r_max * 2))
        R = np.array(R)
        self.P = P
        self.R = R

    def solve_helper(self, gamma, epsilon, p, prob):
        # solve the MDP, with an extra threshold to guarantee value iteration
        # ends if gamma=1
        v, pi = SolveMDP(
            self.P,
            self.R,
            gamma,
            epsilon,
            p,
            prob,
            threshold=self.t_max * self.r_max * 3,
        )

        # store values and policies and matrices
        self.v = v
        self.pi = pi


    # opt_model_trajectory() takes a start state, a transition function,
    # indices of features to be considered, a transition function, and an int
    # for number of points to be plotted. Plots and returns the transitions
    def opt_model_trajectory(
        self,
        x,  # start state as tuple or array
        f,  # transition function of the form f(x, u) = x'
        f1=0,  # index of feature 1 to be plotted
        f2=1,  # index of feature 2 to be plotted
        n=30,
    ):  # points to be plotted

        xs, ys, all_vecs = model_trajectory(self, f, x, f1, f2, n)
        return xs, ys

    # update_predictor
    def update_predictor(self, predictor):
        self.m = predictor
        return
