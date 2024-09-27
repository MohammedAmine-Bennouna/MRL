# MRL

This is the GitHub repo that contains an implementation of the **Minimal Representation Learning Algorithm (MRL)**.

Given a dataset of transitions generated from a continuous state space environment, the model is a heuristic to learn the minimal representation Markov Decision Process (MDP) that preserves the dynamics of the environment. The trained model can then be used to predict values of the system given a starting state and a decision policy or sequence of actions, and can also be used to derive the optimal policy.

For more information about the algorithm, please refer to the paper: **Learning the Minimal Representation of a Dynamic System from Transition Data** https://pubsonline.informs.org/doi/10.1287/mnsc.2022.01652.

## Installing Dependencies 

Please install the libraries in `requirements.txt` with the command:
``bash
`pip install -r requirements.txt` first. 
``

## Repo Content
The model class named `MDP_model` can be imported from the `model.py` file found in the `mrl` folder. Once an instance of this model class is initialized, one can then run `model.fit()` or `model.fit_CV()` to train the model on the dataframe given, without or with cross validation. 

The format of the dataframe inputted needs to have the following columns: 
ID | TIME | FEATURE_1 | FEATURE_2 | ... | ACTION | RISK | OG_CLUSTER (Optional) |
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
0 | 0 | ... | ... | ... | ... | ... | ... | 
0 | 1 | ... | ... | ... | ... | ... | ... | 
... | ... | ... | ... | ... | ... | ... | ... | 

The final `OG_CLUSTER` column is optional, used in the case that there is a "correct" clustering already known for the dataset, for purposes of calculating state classification accuracy after training. 

The `maze/maze_experiment` notebook provides a step by step application of MRL on a Maze environment.