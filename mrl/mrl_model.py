from model import MDP_model
from testing import predict_cluster, training_value_error, get_MDP, next_clusters
from clustering import fit_CV, initializeClusters, splitter
from sklearn.metrics import accuracy_score
import pandas as pd


class MRL_model(MDP_model):
    # fit() takes in the parameters for prediction, and directly fits the model
    # to the data without running cross validation. If optimize is set to True,
    # stores the best clustering in self.df_trained; otherwise stores the
    # clustering at when max_k number of clusters is reached.
    def fit(
        self,
        data,  # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
        # Needs a dataframe where 'ACTION' == 'None' if goal state is reached.
        pfeatures,  # int: number of features
        h=5,  # int: time horizon (# of actions we want to optimize)
        gamma=1,  # discount value
        max_k=70,  # int: max number of clusters
        distance_threshold=0.05,  # clustering diameter for Agglomerative clustering
        cv=5,  # number for cross validation
        th=0,  # splitting threshold
        eta=float("inf"),  # incoherence threshold
        precision_thresh=1e-14,  # precision threshold
        classification="DecisionTreeClassifier",  # classification method
        split_classifier_params={"random_state": 0},  # dict of classifier params
        clustering="Agglomerative",  # clustering method from Agglomerative, KMeans, and Birch
        n_clusters=None,  # number of clusters for KMeans
        random_state=0,
        plot=False,
        optimize=True,
        verbose=False,
    ):

        df = data.copy()

        # save relevant data
        self.df = df
        self.pfeatures = pfeatures
        self.eta = eta
        self.t_max = df["TIME"].max()
        self.r_max = abs(df["RISK"]).max()

        # training on all the data
        df_init = initializeClusters(
            self.df,
            clustering=clustering,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            random_state=random_state,
        )

        # change end state to 'end'
        df_init.loc[df_init["ACTION"] == "None", "NEXT_CLUSTER"] = "End"
        print("Clusters Initialized")
        if verbose:
            print(df_init)

        (
            df_new,
            df_incoherences,
            df_training_error,
            testing_error,
            best_df,
            opt_k,
            split_scores,
            training_error,
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

        self.m = predict_cluster(self.df_trained, self.pfeatures)
        pred = self.m.predict(self.df_trained.iloc[:, 2 : 2 + self.pfeatures])
        self.clus_pred_accuracy = accuracy_score(pred, self.df_trained["CLUSTER"])

        self.df_compare = self.df_trained

        # store P_df and R_df values
        P_df, R_df = get_MDP(self.df_trained)
        self.P_df = P_df
        self.R_df = R_df

        # store next_clusters dataframe
        self.nc = next_clusters(self.df_trained)

    # fit_CV() takes in parameters for prediction, and trains the model on the
    # optimal clustering for a given horizon h (# of actions), using cross
    # validation. See fit_CV in clustering.py for further documentation.
    def fit_CV(
        self,
        data,  # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
        # Needs a dataframe where 'ACTION' == 'None' if goal state is reached.
        pfeatures,  # int: number of features
        h=5,  # int: time horizon (# of actions we want to optimize)
        gamma=1,  # discount value
        max_k=70,  # int: max number of clusters
        distance_threshold=0.05,  # clustering diameter for Agglomerative clustering
        cv=5,  # number of folds for cross validation
        th=0,  # splitting threshold
        eta=float(
            "inf"
        ),  # incoherence threshold, calculated by eta*sqrt(datapoints)/clusters
        precision_thresh=1e-14,  # precision threshold
        classification="DecisionTreeClassifier",  # classification method
        split_classifier_params={"random_state": 0},
        clustering="Agglomerative",  # clustering method from Agglomerative, KMeans, and Birch
        n_clusters=None,  # number of clusters for KMeans
        random_state=0,
        plot=False,
        verbose=False,
    ):

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
            df_training_error,
            testing_error,
            best_df,
            opt_k,
            split_scores,
            training_error,
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

        # store next_clusters dataframe
        self.nc = next_clusters(df_new)
