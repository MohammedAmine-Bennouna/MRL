#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Load Libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import gym
from scipy.stats import binom


import sys

sys.path.append("../mrl/")
from mdp_utils import SolveMDP

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

# list of maze options to choose from:
mazes = {
    1: "maze-v0",
    2: "maze-sample-3x3-v0",
    3: "maze-random-3x3-v0",
    4: "maze-sample-5x5-v0",
    5: "maze-random-5x5-v0",
    6: "maze-sample-10x10-v0",
    7: "maze-random-10x10-v0",
    8: "maze-sample-100x100-v0",
    9: "maze-random-100x100-v0",
    10: "maze-random-10x10-plus-v0",
    11: "maze-random-20x20-plus-v0",
    12: "maze-random-30x30-plus-v0",
}


def truncate(x):
    x2 = x if x < 1 else 0.999
    x3 = x2 if x > 0 else 0.001
    return x3


# createSamples() N, the number of IDs, T_max, the maximum timestep if win-
# state is not reached, and maze (str of maze-name from above dict). Generates
# the samples based on a ratio r of randomness, and (1-r) actions taken according
# to the optimal policy of the maze. If
# reseed = True, selects a random location in the next cell, otherwise reseed=
# p_randomtrans: probability to take a random transition
# False makes robot travel to the next cell with the same offset
# returns a dataframe of the form ['ID', 'TIME', features, 'ACTION', 'RISK']
# if stochastic=True, we use a stochastic maze where after we select the action
# and perform the check for 'r', we also choose a random transition with probability p_randomtrans
def createSamples(
    N,
    T_max,
    maze,
    r,
    p_randomtrans=0,
    reseed=False,
    stochastic=False,
    normal_offset=False,
):
    # solving for optimal policy
    P, R = get_maze_MDP(maze)
    v, pi = SolveMDP(P, R, 0.98, 1e-10, False, "max")

    # initialize environment
    env = gym.make(maze)
    transitions = []
    l = env.unwrapped.maze_size[0]

    for i in range(N):

        # initialize variables
        if normal_offset:
            offset = np.random.normal(0.5, 1 / 6, 2)  # (2**0.5)
            offset = np.array(tuple(offset))
            # offset = np.array( (truncate(np.random.normal(0.5, 1/6)), truncate(np.random.normal(0.5, 1/6))) ) # TODO: test
        else:
            offset = np.array((random.random(), random.random()))
        obs = env.reset()

        # initialize first reward
        reward = -1 / (l * l)
        x = obs + offset
        ogc = int(obs[0] + obs[1] * l)

        for t in range(T_max):
            # take random step or not
            if random.random() <= r:
                action = env.action_space.sample()
            else:
                action = int(pi[ogc])

            if random.random() > p_randomtrans or not stochastic:
                true_action = action
            else:
                true_action = env.action_space.sample()

            transitions.append([i, t, x, action, reward, ogc])

            new_obs, reward, done, info = env.step(true_action)
            ogc = int(new_obs[0] + new_obs[1] * l)

            # if reseed, create new offset
            if reseed:
                if normal_offset:
                    offset = np.random.normal(0.5, 1 / 6, 2)  # (2**0.5)
                    offset = np.array(tuple(offset))
                    offset = np.array(
                        (
                            truncate(np.random.normal(0.5, 1 / 6)),
                            truncate(np.random.normal(0.5, 1 / 6)),
                        )
                    )  # TODO: test

                else:
                    offset = np.array((random.random(), random.random()))

            # if end state reached, append one last no action no reward
            if done:
                transitions.append([i, t + 1, new_obs + offset, "None", reward, ogc])
                break
            x = new_obs + offset

    env.close()

    df = pd.DataFrame(
        transitions, columns=["ID", "TIME", "x", "ACTION", "RISK", "OG_CLUSTER"]
    )

    features = df["x"].apply(pd.Series)
    features = features.rename(columns=lambda x: "FEATURE_" + str(x))

    df_new = pd.concat([df.iloc[:, :2], features, df.iloc[:, 3:]], axis=1)
    df_new["FEATURE_1"] = -df_new["FEATURE_1"]

    return df_new


# opt_model_trajectory() takes a trained model, the maze used to train this model, and
# plots the path of the optimal solution through the maze. returns the path
def opt_model_trajectory(m, maze,max_steps = 200,resolve = False):
    if m.v is None or resolve:
        m.solve_MDP()
    env = gym.make(maze).unwrapped
    obs = env.reset()
    l = env.maze_size[0]
    reward = -1 / (l * l)

    xs = [obs[0]]
    ys = [-obs[1]]
    done = False
    step = 0

    offset = np.array((random.random(), -random.random()))
    point = np.array((obs[0], -obs[1])) + offset

    while (not done) and (step < max_steps):
        step = step + 1
        # find current state and action
        s = m.m.predict(point.reshape(1, -1))
        a = int(m.pi[s])

        obs, reward, done, info = env.step(a)

        offset = np.array((random.random(), -random.random()))
        point = np.array((obs[0], -obs[1])) + offset
        xs.append(obs[0])
        ys.append(-obs[1])

    env.close()

    xs = np.array(xs)
    ys = np.array(ys)

    u = np.diff(xs)
    v = np.diff(ys)
    pos_x = xs[:-1] + u / 2
    pos_y = ys[:-1] + v / 2
    norm = np.sqrt(u ** 2 + v ** 2)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
    # ax.set_xlabel('FEATURE_%i' %f1)
    # ax.set_ylabel('FEATURE_%i' %f2)
    plt.ylim(-l + 0.8, 0.2)
    plt.xlim(-0.2, l - 0.8)
    plt.show()
    return xs, ys


# def policy_trajectory() takes a policy, a maze, and plots the optimal
# trajectory of the policy through the maze, for a total of n steps.
# Can use with fitted_Q policy etc.
def policy_trajectory(policy, maze, n=50, rand=True):
    env = gym.make(maze)
    obs = env.reset()
    l = env.maze_size[0]
    reward = -1 / (l * l)

    offset = np.array((random.random(), -random.random()))
    point = list(np.array((obs[0], -obs[1])) + offset)

    if rand:
        xs = [point[0]]
        ys = [point[1]]
    else:
        xs = [obs[0]]
        ys = [-obs[1]]

    done = False
    i = 0

    while not done:
        # find relevant action
        # print(point)
        a = policy.get_action(point)
        # print(a)
        obs, _, done, info = env.step(a)

        offset = np.array((random.random(), -random.random()))
        point = list(np.array((obs[0], -obs[1])) + offset)

        if rand:
            xs.append(point[0])
            ys.append(point[1])
        else:
            xs.append(obs[0])
            ys.append(-obs[1])

        i += 1
        if i == n:
            done = True

    env.close()

    xs = np.array(xs)
    ys = np.array(ys)

    u = np.diff(xs)
    v = np.diff(ys)
    pos_x = xs[:-1] + u / 2
    pos_y = ys[:-1] + v / 2
    norm = np.sqrt(u ** 2 + v ** 2)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
    if rand:
        plt.ylim(-l - 0.2, 0.2)
        plt.xlim(-0.2, l + 0.2)
    else:
        plt.ylim(-l + 0.8, 0.2)
        plt.xlim(-0.2, l - 0.8)
    plt.show()
    return xs, ys


# policy_accuracy() takes a trained model and a maze, compares every line of
# the original dataset to the real optimal policy and the model's optimal policy,
# then returns the percentage correct from the model
def policy_accuracy(m, maze, df):
    if m.v is None:
        m.solve_MDP(gamma=1)

    # finding the true optimal:
    P, R = get_maze_MDP(maze)
    true_v, true_pi = SolveMDP(P, R, 0.98, 1e-10, True, "max")

    correct = 0
    # iterating through every line and comparing
    for index, row in df.iterrows():
        # predicted action:
        s = m.m.predict(np.array(row[2 : 2 + m.pfeatures]).reshape(1, -1))
        # s = m.df_trained.iloc[index]['CLUSTER']
        a = m.pi[s]

        # real action:
        a_true = true_pi[row["OG_CLUSTER"]]
        if a == a_true:
            correct += 1
    total = df.shape[0]
    return correct / total


# fitted_Q_policy_accuracy() takes a policy given by fitted_Q, the maze,
# and the original dataframe, compares every line of the original dataset to
# the real optimal policy and the fitted_Q's optimal policy, then returns the
# percentage correct from fitted_Q
def fitted_Q_policy_accuracy(policy, maze, df):

    # finding the true optimal:
    P, R = get_maze_MDP(maze)
    true_v, true_pi = SolveMDP(P, R, 0.98, 1e-10, True, "max")

    correct = 0
    # iterating through every line and comparing
    for index, row in df.iterrows():
        # predicted action:
        a = policy.get_action(list(row[2:4]))

        # real action:
        a_true = true_pi[row["OG_CLUSTER"]]
        if a == a_true:
            correct += 1
    total = df.shape[0]
    return correct / total


# opt_maze_trajectory() takes a maze name, then solves the policy and plots the
# optimal path through the maze. Returns the path. ONLY WORKS for deterministic
# mazes!
def opt_maze_trajectory(maze):
    P, R = get_maze_MDP(maze)
    v, pi = SolveMDP(P, R, 0.98, 1e-10, True, "max")

    env = gym.make(maze).unwrapped
    obs = env.reset()
    l = env.maze_size[0]
    reward = -1 / (l * l)

    xs = [obs[0]]
    ys = [-obs[1]]
    done = False

    while not done:
        # find current state and action
        ogc = int(obs[0] + obs[1] * l)
        # print(ogc)
        a = int(pi[ogc])

        obs, reward, done, info = env.step(a)
        # print(done)
        xs.append(obs[0])
        ys.append(-obs[1])

    env.close()

    xs = np.array(xs)
    ys = np.array(ys)

    u = np.diff(xs)
    v = np.diff(ys)
    pos_x = xs[:-1] + u / 2
    pos_y = ys[:-1] + v / 2
    norm = np.sqrt(u ** 2 + v ** 2)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
    # ax.set_xlabel('FEATURE_%i' %f1)
    # ax.set_ylabel('FEATURE_%i' %f2)
    plt.show()
    return xs, ys


# opp_action() returns the opposite action as input action
def opp_action(a):
    if a == 0:
        return 1
    elif a == 1:
        return 0
    elif a == 2:
        return 3
    elif a == 3:
        return 2


# get_maze_MDP() takes a maze string name, and returns two matrices, P and R,
# which describe the MDP of the maze (includes a sink node)
def get_maze_MDP(maze):
    # initialize maze
    env = gym.make(maze)
    obs = env.reset()
    l = env.unwrapped.maze_size[0]

    # initialize matrices
    a = 4
    P = np.zeros((a, l * l + 1, l * l + 1))
    R = np.zeros((a, l * l + 1))
    # P = np.zeros((a, l*l, l*l))
    # R = np.zeros((a, l*l))

    # store clusters seen and cluster/action pairs seen in set
    c_seen = set()
    ca_seen = set()

    # initialize env, set reward of original
    obs = env.reset()
    ogc = int(obs[0] + obs[1] * l)
    reward = -1 / (l * l)

    while len(ca_seen) < (4 * l * l - 4):
        # update rewards for new cluster
        if ogc not in c_seen:
            for i in range(a):
                # R[i, ogc] = reward
                R[i, ogc] = reward
            c_seen.add(ogc)

        stop = False
        for i in range(a):
            if (ogc, i) not in ca_seen:
                ca_seen.add((ogc, i))
                # print(len(ca_seen))
                new_obs, reward, done, info = env.step(i)

                ogc_new = int(new_obs[0] + new_obs[1] * l)
                # update probabilities
                P[i, ogc, ogc_new] = 1
                # print('updated', ogc, ogc_new, done)
                if not done:
                    if ogc != ogc_new:
                        P[opp_action(i), ogc_new, ogc] = 1
                        # print('updated', ogc_new, ogc)
                        ca_seen.add((ogc_new, opp_action(i)))
                    # print(len(ca_seen))
                ogc = ogc_new
                # print('new ogc', ogc, done)

                if done:
                    # set next state to sink
                    for i in range(a):
                        P[i, ogc_new, l * l] = 1
                        P[i, l * l, l * l] = 1
                        R[i, l * l] = 0
                        R[i, ogc_new] = 1
                    obs = env.reset()
                    ogc = int(obs[0] + obs[1] * l)

                stop = True
            if stop:
                break

        # if all seen already, take random step
        if not stop:
            action = env.action_space.sample()
            new_obs, reward, done, info = env.step(action)

            ogc = int(new_obs[0] + new_obs[1] * l)
            # print('trying random action', ogc)
            if done:
                obs = env.reset()
                ogc = int(obs[0] + obs[1] * l)
    env.close()

    return P, R


# Helper function for get_stochastic_maze_MDP
# from a current state in a maze env, acquire all possible next/adjacent states
def get_next_states(env):
    og_state = env.unwrapped.state.copy()
    possible_states = []
    for i in range(4):
        new_state = env.step(i)[0].copy()
        possible_states.append(new_state)
        if not np.array_equal(new_state, og_state):
            env.step(opp_action(i))
    return possible_states


# get_stochastic_maze_MDP() takes a maze string name and stochastic parameter,
# and returns two matrices, P and R, which describe the MDP of the maze (includes a sink node)
def get_stochastic_maze_MDP(maze, p_randomtrans=0.3):
    # initialize maze
    env = gym.make(maze)
    obs = env.reset()
    l = env.unwrapped.maze_size[0]

    # initialize matrices
    a = 4
    P = np.zeros((a, l * l + 1, l * l + 1))
    R = np.zeros((a, l * l + 1))
    # P = np.zeros((a, l*l, l*l))
    # R = np.zeros((a, l*l))

    # store clusters seen and cluster/action pairs seen in set
    c_seen = set()
    ca_seen = set()

    # initialize env, set reward of original
    obs = env.reset()
    ogc = int(obs[0] + obs[1] * l)
    reward = -1 / (l * l)

    while len(ca_seen) < (4 * l * l - 4):
        # update rewards for new cluster
        if ogc not in c_seen:
            for i in range(a):
                R[i, ogc] = reward
            c_seen.add(ogc)

        possible_states = get_next_states(env)
        stop = False
        for i in range(a):
            if (ogc, i) not in ca_seen:
                ca_seen.add((ogc, i))
                # print(len(ca_seen))

                new_obs, reward, done, info = env.step(i)

                ogc_new = int(new_obs[0] + new_obs[1] * l)

                # update probabilities
                P[i, ogc, ogc_new] += 1 - p_randomtrans
                for next_state in possible_states:
                    next_cluster = int(next_state[0] + next_state[1] * l)
                    P[i, ogc, next_cluster] += p_randomtrans / 4

                # update in other direction
                """if not done:
                    if ogc != ogc_new and (ogc_new, opp_action(i)) not in ca_seen:
                        ca_seen.add((ogc_new, opp_action(i)))
                        P[opp_action(i), ogc_new, ogc] += (1-p_randomtrans) + p_randomtrans/4
                        possible_states2 = get_next_states(env)
                        for next_state2 in possible_states2: 
                            if next_state2 == 
                            next_cluster2 = int(next_state[0] + next_state[1]*l)
                            P[opp_action(i), ogc_new, next_cluster2] += p_randomtrans/4"""

                # update current cluster/position
                ogc = ogc_new
                if done:
                    # set next state to sink
                    for i in range(a):
                        P[i, ogc_new, l * l] = 1
                        P[i, l * l, l * l] = 1
                        R[i, l * l] = 0
                        R[i, ogc_new] = 1
                    obs = env.reset()
                    ogc = int(obs[0] + obs[1] * l)

                stop = True
            if stop:
                break

        # if all seen already, take random step
        if not stop:
            action = env.action_space.sample()
            new_obs, reward, done, info = env.step(action)

            ogc = int(new_obs[0] + new_obs[1] * l)
            # print('trying random action', ogc)
            if done:
                obs = env.reset()
                ogc = int(obs[0] + obs[1] * l)
    env.close()

    return P, R


# get_maze_transition_reward() takes a maze name, and returns the transition function
# in the form of f(x, u) = x'. State, action, gives next state. Takes into account
# sink node, and stays there. Also returns reward function r(x) that takes a state
# and returns the reward
def get_maze_transition_reward(maze):

    P, R = get_maze_MDP(maze)
    l = int((R.size / 4 - 1) ** 0.5)

    def f(x, u, plot=True):
        assert type(x) == tuple or x.shape == (2,), "bad shape"
        assert type(u) == int or u == "None" or u.size == 1, "bad shape"
        # print('x', x, 'u', u)

        # if sink, return None again
        if plot:
            if x[0] == None:
                return (None, None)

        # if no action, or an action '4' to simulate no action:
        if u == "None" or u == 4:
            if plot:
                return (None, None)
            else:
                return (0, -l)

        # first define the cluster of the maze based on position
        x_orig = (int(x[0]), int(-x[1]))
        offset = np.array((random.random(), -random.random()))

        c = int(x_orig[0] + x_orig[1] * l)
        c_new = P[u, c].argmax()

        # if maze at sink, return None
        if c_new == R.size / 4 - 1:
            if plot:
                return (None, None)
            else:
                return (0, -l)
        else:
            x_new = (c_new % l, c_new // l)
            x_new = (x_new[0], -x_new[1])
        return x_new + offset

    def r(x):
        # if sink, return 0 reward
        assert type(x) == tuple or x.shape == (2,), "bad shape"
        if x[0] == None:
            return R[0][-1]
        else:
            x_orig = (int(x[0]), int(-x[1]))
            c = int(x_orig[0] + x_orig[1] * l)
            # print(c)
            return R[0][c]

    return f, r


# get_maze_transition_reward() takes a maze name, and returns the transition function
# in the form of f(x, u) = x'. State, action, gives next state. Takes into account
# sink node, and stays there. Also returns reward function r(x) that takes a state
# and returns the reward
def get_stochastic_maze_transition_reward(maze, p_randomtrans=0.3, normal_offset=False):

    # First get Stoch MDP
    P, R = get_stochastic_maze_MDP(maze, p_randomtrans)
    l = int((R.size / 4 - 1) ** 0.5)

    def f(x, u, plot=True):
        assert type(x) == tuple or x.shape == (2,), "bad shape"
        assert type(u) == int or u == "None" or u.size == 1, "bad shape"
        # print('x', x, 'u', u)

        # if sink, return None again
        if plot:
            if x[0] == None:
                return (None, None)

        # if no action, or an action '4' to simulate no action:
        if u == "None" or u == 4:
            if plot:
                return (None, None)
            else:
                return (0, -l)

        # first define the cluster of the maze based on position
        x_orig = (int(x[0]), int(-x[1]))
        if normal_offset:
            x1 = np.random.normal(0.5, 1 / 6)
            x2 = np.random.normal(0.5, 1 / 6)
            offset = np.array((truncate(x1), -truncate(x2)))
        else:
            offset = np.array((random.random(), -random.random()))

        c = int(x_orig[0] + x_orig[1] * l)
        # simulate stochasticity
        c_new = random.choices(range(len(P[u, c])), weights=P[u, c])[
            0
        ]  # c_new = P[u, c].argmax()

        # if maze at sink, return None
        if c_new == R.size / 4 - 1:
            if plot:
                return (None, None)
            else:
                return (0, -l)
        else:
            x_new = (c_new % l, c_new // l)
            x_new = (x_new[0], -x_new[1])
        return x_new + offset

    def r(x):
        # if sink, return 0 reward
        assert type(x) == tuple or x.shape == (2,), "bad shape"
        if x[0] == None:
            return R[0][-1]
        else:
            x_orig = (int(x[0]), int(-x[1]))
            c = int(x_orig[0] + x_orig[1] * l)
            # print(c)
            try:
                reward = R[0][c]
                return reward
            except Exception as e:
                print(f"positive coords: {x_orig}")
                raise e

    return f, r


# value_est() takes a list of models or policies, integer K number of simulations,
# real transition matricies P, R and transition/reward functions f and r
# and compares the v_opt (correctly solved MDP value from cell 0) and v_alg
# (value determined by model/policy) for K number of random points from the same
# start cell. Result truncates everything greater than 1 to 1.


def value_est(models, Ns, K, P, R, f, r, true_v=None, true_pi=None):
    # first calculate v_opt for this particular maze and t_max steps
    v_opt = 0
    s = 0
    if true_v is None or true_pi is None:
        true_v, true_pi = SolveMDP(P, R, prob="max", gamma=1, epsilon=1e-8)
    # print(true_v)
    v_opt = true_v[0]

    # then for each model and policy, run through the K trials and return
    # an array of differences corresponding to each N
    n = len(Ns)

    v_alg = []
    # for each n of this set:
    for i in range(n):
        # print('Round N=', Ns[i])
        m = models[i]

        # calculating average value for this model and policy
        if m.pi is None:
            m.solve_MDP()

        model_vs = []
        for k in range(K):
            start = np.array((random.random(), -random.random()))
            s = m.m.predict(start.reshape(1, -1))
            val = m.v[s]
            # print('val', val, 'v_opt', v_opt, 's', s)
            v_diff = abs(val - v_opt)
            val_trunc = min(1, v_diff)  # truncate and take 1 as highest

            if isinstance(val_trunc, np.ndarray):
                val_trunc = val_trunc[0]
            # print(val_trunc)
            model_vs.append(val_trunc)

        model_v = np.mean(model_vs)
        # print('model avg val', model_v)
        v_alg.append(model_v)

    # calculate differences between values and optimal
    # v_alg_diff = abs(v_alg - v_opt)
    return v_alg


# plot_paths() takes a dataframe with 'FEATURE_1' and 'FEATURE_2', and plots
# the first n paths (by ID). Returns nothing
def plot_paths(df, n,size=5):
    fig, ax = plt.subplots()

    for i in range(n):
        x = df.loc[df["ID"] == i]["FEATURE_0"]
        y = df.loc[df["ID"] == i]["FEATURE_1"]
        xs = np.array(x)
        ys = np.array(y)

        u = np.diff(xs)
        v = np.diff(ys)
        pos_x = xs[:-1] + u / 2
        pos_y = ys[:-1] + v / 2
        norm = np.sqrt(u ** 2 + v ** 2)

        ax.plot(xs, ys, marker="o")
        ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")

    # Set fixed ticks for the x-axis
    xticks = [k for k in range(size+1)]
    plt.xticks(xticks)
    yticks = [-k for k in range(size+1)]
    plt.yticks(yticks)
    plt.grid(which="both", axis="both", linestyle="--", color="gray")
    plt.show()
    return


# fitted_Q() trains K functions Q1 to QK that determine the optimal strategy
# x_df is a dataframe of the form ['ID', 'TIME', features, 'ACTION', 'RISK']
# for each one-step transition returns the last iteration QK, and a function \
# policy that takes a state and outputs the optimal action
def fitted_Q(
    K,  # number of iterations
    x_df,  # dataframe
    gamma,  # decay factor
    pfeatures,  # number of features in dataframe
    actions,  # list of action possibilities
    reward_c,  # reward function to initialze Q (can initial to 0, but check types)
    take_max=True,  # True if max_cost is good, otherwise false
    regression="LinearRegression",  # str: type of regression
    one_hot=True,  # one-hot encode the given actions
):

    x_df = x_df.copy(deep=True)
    # x_df = x_df.loc[x_df['']]
    # initialize storage and actions
    # Qs = []
    values = []
    num_actions = len(actions)

    # create the first Q1 function
    class Q:
        def predict(self, array):
            x = array[0][:pfeatures]
            # print('x', x)
            return reward_c(x)

    Q_new = Q()
    # Qs.append(Q_new)

    # calculate the x_t2 next step for each x
    x_df["x_t2"] = x_df.iloc[:, 2 : 2 + pfeatures].values.tolist()
    x_df["x_t2"] = x_df["x_t2"].shift(-1)

    # delete the ones that don't finish
    x_df.loc[
        (x_df["ID"] != x_df["ID"].shift(-1)) & (x_df["ACTION"] != "None"), "ACTION"
    ] = "replace"
    x_df = x_df[x_df["ACTION"] != "replace"]

    # deal with last x_t2 if it happens to finish, with empty list
    x_df.loc[x_df["x_t2"].isnull(), ["x_t2"]] = x_df.loc[
        x_df["x_t2"].isnull(), "x_t2"
    ].apply(lambda x: [])

    if one_hot:
        actions_new = [
            [int(x == i) for x in range(num_actions)] for i in range(num_actions)
        ]

    else:
        actions_new = [[a] for a in actions]

    for i in range(num_actions):
        # regular action names
        x_df["a%s" % i] = x_df.apply(lambda x: x.x_t2 + actions_new[i], axis=1)

    action_names = ["a%s" % i for i in range(num_actions)]

    print(x_df, flush=True)
    # create X using x_t and u
    # select (x_t, u) pair as training
    # setting 'None' action as action -1

    if one_hot:
        X = x_df.iloc[:, 2 : 2 + pfeatures]
        one_hot_actions = pd.get_dummies(x_df["ACTION"])[actions]
        # represent 'None' as action 0 in the training data so it gets predicted
        try:
            one_hot_actions.loc[one_hot_actions["None"] == 1, 0] = 1
        except:
            pass
        one_hot_actions = one_hot_actions[actions]
        X = pd.concat([X, one_hot_actions], axis=1)
    else:
        X = x_df.iloc[:, 2 : 3 + pfeatures]
        X.loc[X["ACTION"] == "None", "ACTION"] = 0

    # maybe trying to put action as a tuple of (0, 1) or 1-hot to help it learn better...?
    # X = x_df[['FEATURE_0', 'FEATURE_1']].merge(x_df['ACTION'].apply(pd.Series), \
    # left_index = True, right_index = True)

    print("New training features", flush=True)
    print(X, flush=True)

    # update function that returns the correct updated Q for the given row
    # and the Q_new prediction function. Takes just the risk (with no extra update)
    # for ending state, which is when ACTION = 'None'
    def update(row, Q_new, take_max):
        # print('action:', row.ACTION)
        if row.ACTION == "None":
            # print('returning this')
            return np.array([row.RISK])
        if take_max:
            return row.RISK + gamma * max(
                [Q_new.predict([g]) for g in [row[a] for a in action_names]]
            )
        else:
            return row.RISK + gamma * min(
                [Q_new.predict([g]) for g in [row[a] for a in action_names]]
            )

    bar = tqdm(range(K))
    # bar = range(K)
    # creating new Qk functions
    for i in bar:
        # create y using Qk-1 and x_t2
        # non-DP
        """
        if take_max: 
            y = x_df.apply(lambda x: x.RISK + gamma*max([Q_new.predict([g]) \
                                for g in [x[a] for a in action_names]]), axis=1)
        else:
            y = x_df.apply(lambda x: x.RISK + gamma*min([Q_new.predict([g]) \
                                for g in [x[a] for a in action_names]]), axis=1)
        """
        y = x_df.apply(lambda x: update(x, Q_new, take_max), axis=1)

        """                               
        # initialize dp
        memo = {}
        mu = 0
        y = []                        
        for index, row in x_df.iterrows():
            qs = []
            for f in [row['a0'], row['a1'], row['a2'], row['a3']]:
                if f in memo:
                    qs.append(memo[f])
                    #print('memo used')
                    mu += 1
                else:
                    q = Q_new.predict([f])
                    memo[f] = q
                    qs.append(memo[f])
            y.append(row['c'] + gamma*min(qs))
        """

        y = np.array(y)
        # print(y, flush=True)
        # print(np.unique(y), flush=True)
        # print(y)

        # train the actual Regression function as Qk
        # regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
        if regression == "LinearRegression":
            regr = LinearRegression().fit(X, y)
        if regression == "RandomForest":
            regr = RandomForestRegressor(random_state=0, oob_score=True).fit(X, y)
        if regression == "ExtraTrees":
            regr = ExtraTreesRegressor(n_estimators=50).fit(X, y.ravel())
        if regression == "SGDRegressor":
            regr = SGDRegressor().fit(X, y.ravel())
        if regression == "DecisionTrees":
            regr = DecisionTreeRegressor(random_state=5).fit(X, y.ravel())
        # Qs.append(regr)
        Q_new = regr
        # print('memo size', len(memo), 'used', mu, flush=True)

        # calculate and store value of new policy
        p = policy(actions_new, take_max)
        p.fit(Q_new)
        v = p.get_value([0.5, -0.5])  # get value of starting state
        a = p.get_action([0.5, -0.5])
        values.append(v)

        print("Iteration:", i, "Start Value", v, "action", a)

    # QK = Qs[-1]
    QK = Q_new

    p = policy(actions_new, take_max)
    p.fit(QK)

    return QK, p, x_df, values


class policy:
    def __init__(self, actions, take_max):
        self.QK = None
        self.actions = actions
        self.take_max = take_max

    def fit(self, QK):  # model, the latest fitted_Q
        self.QK = QK

    # get_action() takes a state x and predicts the optimal action
    def get_action(self, x):
        if self.take_max:
            i = np.argmax([self.QK.predict([x + u]) for u in self.actions])
            # print([self.QK.predict([x + [u]]) \
            # for u in self.actions])
        else:
            i = np.argmin([self.QK.predict([x + u]) for u in self.actions])

        if len(self.actions[i]) == 1:
            return self.actions[i]
        else:
            return np.argmax(self.actions[i])

    # get_value() takes a state x and returns the max value from fitted Q
    def get_value(self, x):
        if self.take_max:
            val = max([self.QK.predict([x + u]) for u in self.actions])
        else:
            val = min([self.QK.predict([x + u]) for u in self.actions])
        return val



def value_diff(
    models, #models to evaluate
    Ns,
    K,
    t_max,
    P,
    R,
    f,
    r,
    take_max=True,
    gamma=0.99,
    true_v=None,
    true_pi=None,
    policy=False,
    normal_offset=False,
    mu=0.5,
    std=1 / 6,
    mean=True,
):

    '''
    value_diff() takes a list of models or policies, a list
    of data sizes N from which models/policies were trained (same length as list
    of models/policies), parameter K number of simulations from starting cell,
    the real P, R transition matrices, and f, r model transition function and rewards.
    calculates optimality gap difference between values: |v_policy/algo - v_opt*|.
    v_policy/algo is found by randomly generating K points in the starting cell,
    simulating over t_max steps through the actual maze, and taking the avg
    over these K trials. v_opt is the value taking the optimal policy
    If toggle policy = True, calculates the optimality gap of a list of policies
    fed in as the first parameter in the 'models' input.
    if the parameter 'mean is false, we will use the median to compute the value diff.
    '''

    l = 5  #TODO: maze.unwrapped.size

    def truncate(x):
        # truncates a value to make it between 0 and 1
        x2 = x if x < 1 else 0.999
        x3 = x2 if x > 0 else 0.001
        return x3

    def get_normal_offset(mean=0.5, std=1 / 6):
        # gets a normal offset with mean 0.5 standard deviation 1/6
        x1 = np.random.normal(mean, std)
        x2 = np.random.normal(mean, std)
        offset = np.array((truncate(x1), -truncate(x2)))
        return offset

    for model in models:
        if model.pi is None:
            model.solve_MDP(gamma=1, epsilon=1e-4)

    if true_pi is None:
        true_v, true_pi = SolveMDP(P, R, prob="max", gamma=1, epsilon=1e-8)
    
    
    n = len(Ns)

    v_alg = []
 
    # for each n of this set:
    for i in range(n):
        # print("Round N=", Ns[i])
        mod = models[i]

        # model_vs = []
        v_estims = []
        v_opts = []
        for k in range(K):
            # initializing model value estimates
            vm_estim = 0  # under the learned model policy
            v_opt_test = 0  # under the optimal maze policy

            # initialize starting positions
            if normal_offset:
                x_model = get_normal_offset(mu, std)
                x_opt = get_normal_offset(mu, std)
            else:
                x_model = np.array((random.random(), -random.random()))
                x_opt = np.array((random.random(), -random.random()))

            # estimate value for model or policy
            vm_estim += r(x_model)
            v_opt_test += r(x_opt)
            for t in range(t_max):
                # predict action and upate value for model
                if x_model[0] == None:
                    a = 0
                else:
                    s = int(mod.m.predict([x_model]))
                    a = mod.pi[s]

                if x_opt[0] == None:
                    a_opt = 0
                else:
                    x_orig = (int(x_opt[0]), int(-x_opt[1]))
                    ogc = int(x_orig[0] + x_orig[1] * l)
                    a_opt = true_pi[ogc]

                # Get next state
                x_model_new = f(x_model, a)
                x_opt_new = f(x_opt, a_opt)
                # record reward
                vm_estim += r(x_model_new)
                v_opt_test += r(x_opt_new)
                # Update state
                x_model = x_model_new
                x_opt = x_opt_new

            # append the total value from this trial
            v_estims.append(vm_estim)
            v_opts.append(v_opt_test)

        # average values of all trials for this model/policy
        if mean:
            model_v = abs(np.mean(v_estims) - np.mean(v_opts))
        else:
            model_v = abs(np.median(v_estims) - np.median(v_opts))
        v_alg.append(model_v)

    return v_alg


# opt_path_value_diff() compares the v_opt from the optimal sequence of actions
# to the value derived from the MDP also taking this exact sequence of actions
# averaged over K trials. Result truncates anything greater than 1 to 1.
# Takes a list of models/policies, list of data sizes used to train models Ns,
# int K number of simulations, real transition matrices P, R and real transition
# and reward functions f, r from get_maze_transition_reward
# Returns list of differences for each model
def opt_path_value_diff(models, Ns, K, t_max, P, R, f, r, true_v=None, true_pi=None):
    # first calculate v_opt for this particular maze and t_max steps
    v_opt = 0
    s = 0
    actions = []
    if true_v is None or true_pi is None:
        true_v, true_pi = SolveMDP(P, R, prob="max", gamma=1, epsilon=1e-8)
    for t in range(t_max):
        v_opt += R[0, s]
        # print(R[0, s], v_opt)
        a = true_pi[s]
        actions.append(a)
        s_new = P[a, s].argmax()
        s = s_new
        # print(s)

    # then for each model or policy, run through the K trials and return
    # an array of differences corresponding to each N
    n = len(Ns)

    v_alg = []

    # for each n of this set:
    for i in range(n):
        # print('Round N=', Ns[i])
        m = models[i]

        # calculating average value for this model
        if m.pi is None:
            # print('resolved model')
            m.solve_MDP(gamma=1, epsilon=1e-4)
        # m.solve_MDP(gamma=0.999)

        model_vs = []

        for k in range(K):
            vm_estim = 0  # initializing model value estimate

            start = np.array((random.random(), -random.random()))

            # estimate value for model / policy

            s = m.m.predict(start.reshape(1, -1))
            # s_prime = s
            vm_estim += m.R[0, s]

            # take the list of actions through MDP and update value
            for a in actions:

                s = m.P[a, s].argmax()
                vm_estim += m.R[0, s]

            # append the total value from this trial
            vm_diff = abs(vm_estim - v_opt)
            val_trunc = min([1], vm_diff)
            model_vs.append(val_trunc)

        # average values of all trials for this model/policy
        model_v = np.mean(model_vs)
        v_alg.append(model_v)

    return v_alg



class Maze_Model_Visualizer:
    #DEBUG
    def __init__(self, m):
        self.m = m
        self.fig = None

    def plot_features(self, f1='FEATURE_0', f2='FEATURE_1', size=5, s=5, title='MRL Learnt Representation', rep='CLUSTER',cmap = 'tab20', show_legend = False):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Scatter plot with larger point size for better visibility
        scatter = ax.scatter(self.m.df_trained[f1], self.m.df_trained[f2], 
                            c=self.m.df_trained[rep], cmap=cmap, s=s)

        # Function to handle hover events
        def hover(event):
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    # Correctly access the index for hovering points
                    index = ind["ind"][0]  # Get the first point under the cursor
                    cluster = self.m.df_trained.iloc[index]['CLUSTER']
                    ax.set_title(f"Cluster: {cluster}")
                else:
                    ax.set_title("")  # Reset title if no point is hovered
            else:
                ax.set_title("")  # Reset title if out of axes

            fig.canvas.draw_idle()  # Update canvas

        # Connect the hover event to the figure
        fig.canvas.mpl_connect("motion_notify_event", hover)

        # Set x and y labels
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title(title)
        if show_legend:
            plt.legend(title='Clusters', loc='upper right', labels=[str(i) for i in range(self.m.opt_k)])
        # Adjust ticks (optional)
        xticks = [k for k in range(size+1)]
        plt.xticks(xticks)
        yticks = [-k for k in range(size+1)]
        plt.yticks(yticks)
        ax.set_ylim(-size - 0.2, 0.2)
        ax.set_xlim(-0.2, size + 0.2)
        self.fig = ax

        # plt.show()
        
    def show_cluster(self, cluster,color='black',s=10):
        if cluster == self.m.opt_k:
            print('Cluster', cluster,' is an artificial sink state (where paths end), no data points.')
        elif cluster == self.m.opt_k+1:
            print('Cluster', cluster,' is an artificial -\infty reward state (encoding undefined actions), no data points.')
        elif cluster > self.m.opt_k+1:
            raise ValueError('Cluster', cluster,' does not exist, number of clusters is', self.m.opt_k)
        xs = self.m.df_trained[self.m.df_trained['CLUSTER'] == cluster]['FEATURE_0']
        ys = self.m.df_trained[self.m.df_trained['CLUSTER'] == cluster]['FEATURE_1']
        if self.fig is None:
            self.plot_features()
        
        self.fig.scatter(xs, ys, s=s, color=color, marker = '^')
              
    def show_point(self, point, color='red',s=10):
        if self.fig is None:
            self.plot_features()
        self.fig.scatter(point[0], point[1], s=s, color=color, marker='x')

    def policy(self, cluster = None):
        if cluster is not None and cluster > self.m.opt_k:
            raise ValueError('Cluster does not exist, number of clusters is', self.m.opt_k)
        if self.fig is None:
            self.plot_features()
        
        clusters = [c for c in range(self.m.opt_k)] if cluster is None else [cluster]
        for c in clusters:
            xs = self.m.df_trained[self.m.df_trained['CLUSTER'] == c]['FEATURE_0']
            ys = self.m.df_trained[self.m.df_trained['CLUSTER'] == c]['FEATURE_1']
            barycenter_x = sum(xs) / len(xs)
            barycenter_y = sum(ys) / len(ys)
            action = self.m.pi[c]
            direction = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}
            self.fig.arrow(barycenter_x, barycenter_y, 0.7*direction[action][0], 0.7*direction[action][1], head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    

    def show_transition(self, cluster, action,s=10):
        actions = {0: 'Up', 1: 'Down', 2: 'Right', 3: 'Left'}
        print(f'Action {action} corresponds to moving {actions[action]}')
        if cluster > self.m.opt_k:
            raise ValueError('Cluster', cluster,' does not exist, number of clusters is', self.m.opt_k)
        try:
            cluster2 = np.where(self.m.P[action,cluster,:])[0][0]
        except:
            raise ValueError('No constructed transition matrix. Run first m.Solve_MDP()')
        self.show_cluster(cluster,s=s)
        if cluster2 == self.m.opt_k:
            print('End of paths. No transitions defined for this cluster.')
        elif cluster2 == self.m.opt_k+1:
            print('(!)No transition defined for action',action, 'from cluster',cluster,'. The constructed MDP lead it to a -\infty reward state', '(cluster',cluster2,').')
            print('This means it did not pass the robustness filters, or the transition was never oberseved in the data.')   
        else:
            print('Transition is: ',cluster,'--(action',action,')-->',cluster2)
            self.show_cluster(cluster2,color='red',s=s)
        
    def transition_details(self,cluster, action, beta=0.85):
        print('------Transition details for cluster',cluster,'and action',action,'------')
        cluster2 = np.where(self.m.P[action,cluster,:])[0][0]
        if cluster2 == self.m.opt_k:
            print('No transition defined for action',action, 'from cluster',cluster,'. The constructed MDP lead it to a sink state', '(cluster',cluster2,'). This means it did not pass the robustness filters, or the transition was never oberseved in the data.')   
        else:
            print('Transition is: ',cluster,'--(action',action,')-->',cluster2)
        print('>Number of (s=',cluster,',a=',action,') pairs seen:', self.m.nc.loc[cluster,action]['count'])
        print('>Purity: ratio of these going to the majority transition:', self.m.nc.loc[cluster,action]['purity'])
        ncc = self.m.nc.loc[cluster,action]
        print(f'p value that {beta*100}% points of cluster {cluster} taking action {action} going to cluster {ncc["NEXT_CLUSTER"]}:  {(1 - binom.cdf(ncc["purity"] * (ncc["count"]),ncc["count"], beta))}')
        print('------------------------------------------------------')

    def cluster_details(self,cluster):
        print('------Cluster details for cluster',cluster,'------')
        print('Cluster size:', self.m.df_trained[self.m.df_trained['CLUSTER'] == cluster].shape[0])
        if self.m.pi is not None:
            print('Optimal action:', self.m.pi[cluster])
        print('------------------------------------------------------')
    
    def simulate_opt_policy(self, maze,max_steps = 200,show_state_transitions = False):
        if self.m.pi is None:
            raise ValueError('Model has not been trained yet. Please train the model first. Use m.Solve_MDP()')
        env = gym.make(maze).unwrapped
        obs = env.reset()
        l = env.maze_size[0]
        reward = -1 / (l * l)

        xs = [obs[0]]
        ys = [-obs[1]]
        
        done = False
        step = 0

        offset = np.array((random.random(), -random.random()))
        point = np.array((obs[0], -obs[1])) + offset

        x_points = [obs[0]]
        y_points = [-obs[1]]

        while (not done) and (step < max_steps):
            step = step + 1
            # find current state and action
            s = self.m.m.predict(point.reshape(1, -1))
            a = int(self.m.pi[s])

            obs, reward, done, info = env.step(a)

            offset = np.array((random.random(), -random.random()))
            point = np.array((obs[0], -obs[1])) + offset

            x_points.append(point[0])
            y_points.append(point[1])
            xs.append(obs[0])
            ys.append(-obs[1])

        env.close()

        xs,ys = np.array(xs),np.array(ys)
        x_points,y_points = np.array(x_points),np.array(y_points)

        u,v = np.diff(xs),np.diff(ys)
        u_points,v_points = np.diff(x_points),np.diff(y_points)
        pos_x,pos_y = xs[:-1] + u / 2,ys[:-1] + v / 2
        norm = np.sqrt(u ** 2 + v ** 2)

        pos_x_points,pos_y_points = x_points[:-1] + u_points / 2,y_points[:-1] + v_points / 2
        norm_points = np.sqrt(u_points ** 2 + v_points ** 2)

        fig, ax = plt.subplots()
        if show_state_transitions:
            ax.plot(xs, ys, marker="o")
            ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
        else:
            ax.plot(x_points, y_points, marker="o")
            ax.quiver(pos_x_points, pos_y_points, u_points / norm_points, v_points / norm_points, angles="xy", zorder=5, pivot="mid")
        # ax.set_xlabel('FEATURE_%i' %f1)
        # ax.set_ylabel('FEATURE_%i' %f2)
        plt.ylim(-l - 0.2, 0.2)
        plt.xlim(-0.2, l + 0.2)
        plt.title('Optimal Policy Simulation')
        plt.show()
        
#   def plot_features(df, x, y, c="CLUSTER", ax=None, title=None, cmap="tab20", size=5):
#     if ax == None:
#         fig, ax = plt.subplots()

#     df.plot(kind="scatter", x=x, y=y, c=c, cmap=cmap, ax=ax, s=3)

#     ax.set_xlabel(x)
#     ax.set_ylabel(y)
#     # xticks = [0, 1, 2, 3, 4, 5]
#     xticks = [k for k in range(size+1)]
#     plt.xticks(xticks)
#     yticks = [0, -1, -2, -3, -4, -5]
#     yticks = [-k for k in range(size+1)]
#     plt.yticks(yticks)
#     if title != None:
#         plt.title(title)