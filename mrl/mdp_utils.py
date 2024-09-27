# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 22:27:02 2020

@author: omars
"""

#################################################################
# import mdptoolbox, mdptoolbox.example
# from gurobipy import *
import numpy as np
import scipy.sparse as _sp
from numpy import random
from random import choices

#################################################################


#################################################################
# Tools for last function
def _randDense(states, actions, mask):
    # definition of transition matrix : square stochastic matrix
    P = np.zeros((actions, states, states))
    # definition of reward matrix (values between -1 and +1)
    R = np.zeros((actions, states, states))
    for action in range(actions):
        for state in range(states):
            # create our own random mask if there is no user supplied one
            if mask is None:
                m = np.random.random(states)
                r = np.random.random()
                m[m <= r] = 0
                m[m > r] = 1
            elif mask.shape == (actions, states, states):
                m = mask[action][state]  # mask[action, state, :]
            else:
                m = mask[state]
            # Make sure that there is atleast one transition in each state
            if m.sum() == 0:
                m[np.random.randint(0, states)] = 1
            P[action][state] = m * np.random.random(states)
            P[action][state] = P[action][state] / P[action][state].sum()
            R[action][state] = m * (
                2 * np.random.random(states) - np.ones(states, dtype=int)
            )
    return (P, R)


def _randSparse(states, actions, mask):
    # definition of transition matrix : square stochastic matrix
    P = [None] * actions
    # definition of reward matrix (values between -1 and +1)
    R = [None] * actions
    for action in range(actions):
        # it may be more efficient to implement this by constructing lists
        # of rows, columns and values then creating a coo_matrix, but this
        # works for now
        PP = _sp.dok_matrix((states, states))
        RR = _sp.dok_matrix((states, states))
        for state in range(states):
            if mask is None:
                m = np.random.random(states)
                m[m <= 2 / 3.0] = 0
                m[m > 2 / 3.0] = 1
            elif mask.shape == (actions, states, states):
                m = mask[action][state]  # mask[action, state, :]
            else:
                m = mask[state]
            n = int(m.sum())  # m[state, :]
            if n == 0:
                m[np.random.randint(0, states)] = 1
                n = 1
            # find the columns of the vector that have non-zero elements
            nz = m.nonzero()
            if len(nz) == 1:
                cols = nz[0]
            else:
                cols = nz[1]
            vals = np.random.random(n)
            vals = vals / vals.sum()
            reward = 2 * np.random.random(n) - np.ones(n)
            PP[state, cols] = vals
            RR[state, cols] = reward
        # PP.tocsr() takes the same amount of time as PP.tocoo().tocsr()
        # so constructing PP and RR as coo_matrix in the first place is
        # probably "better"
        P[action] = PP.tocsr()
        R[action] = RR.tocsr()
    return (P, R)


def rand(S, A, is_sparse=False, mask=None):
    # making sure the states and actions are more than one
    assert S > 1, "The number of states S must be greater than 1."
    assert A > 1, "The number of actions A must be greater than 1."
    if is_sparse:
        P, R = _randSparse(S, A, mask)
    else:
        P, R = _randDense(S, A, mask)
    return (P, R)


#################################################################


#################################################################
def Generate_random_MDP(
    n,  # State space size
    m,  # Nb of actions
    reduced=True,  # Reward of the form R[a,i] if true, R[a,i,j] if false
    reward_dep_action=False,  # Reward of the form R[i] if false, see reduced else
    deterministic=False,
):
    # n: number of states
    # m: number of actions
    # reduced: if the reward does not depend on the next state i.e R is of the form R(a,i) and not R(a,i,j)
    # Returns - P[a,i,j] probabilty of going to j from i when action is taken
    #         - R[a,i] reward when going to any j from i when action is taken if reduced is true OR R[a,i,j] reward when going to j from i when action is taken if reduced is false
    P, R = rand(n, m)
    if deterministic:
        for s in range(n):
            for a in range(m):
                u = np.argmax(P[a, s, :])
                for sp in range(n):
                    P[a, s, sp] = sp == u
    if reduced:
        Rp = np.zeros((m, n))
        for a in range(m):
            for i in range(n):
                Rp[a, i] = np.abs(np.mean(R[a, i, :]))
        if reward_dep_action == False:
            Rpp = np.zeros(n)
            for i in range(n):
                Rpp[i] = Rp[0, i]
            return P, Rpp
        else:
            return P, Rp
    else:
        return P, R


#################################################################


#################################################################
# Tools for last function
def expand(R, n, m):
    Rp = np.zeros((m, n, n))
    for a in range(m):
        for i in range(n):
            for j in range(n):
                if len(R.shape) == 2:
                    Rp[a, i, j] = R[a, i]
                elif len(R.shape) == 1:
                    Rp[a, i, j] = R[i]
    return Rp


# Bellman operator
def Bell(V, P, R, gamma, prob="min", reduced=True):
    m, n, n = P.shape
    res = np.zeros(n)
    v = V.copy()
    if prob == "min":
        for i in range(n):
            res[i] = sum(P[0, i, :] * (R[0, i, :] + gamma * v))
            for a in range(m):
                res[i] = min(res[i], sum(P[a, i, :] * (R[a, i, :] + gamma * v)))
    if prob == "max":
        for i in range(n):
            res[i] = sum(P[0, i, :] * (R[0, i, :] + gamma * v))
            for a in range(m):
                res[i] = max(res[i], sum(P[a, i, :] * (R[a, i, :] + gamma * v)))
    return res


# Value Iteration
def ValueIteration(
    P, R, gamma=0.9, epsilon=10 ** (-10), prob="min", threshold=float("inf")
):
    m, n, n = P.shape
    V = np.zeros(n)
    W = Bell(V, P, R, gamma, prob)
    while np.linalg.norm(V - W) > epsilon:
        V = W
        W = Bell(W, P, R, gamma, prob)
        if (
            gamma == 1 and max(abs(V)) > threshold
        ):  # threshold in case the value is actually infinity, used when gamma=1
            return V
    return W


# Evaluate a policy in an MDP. WARNING:works only with R[s]
def policy_value(mu, P, R, gamma=0.9, epsilon=10 ** (-10)):
    m, n, n = P.shape
    V = np.ones(n)
    W = V.copy()
    for s in range(n):
        V[s] = R[s] + gamma * sum(
            mu[s, a] * sum(P[a, s, sp] * V[sp] for sp in range(n)) for a in range(m)
        )
    while np.linalg.norm(V - W) > epsilon:
        W = V.copy()
        for s in range(n):
            V[s] = R[s] + gamma * sum(
                mu[s, a] * sum(P[a, s, sp] * V[sp] for sp in range(n)) for a in range(m)
            )
    return V


# Getting Policy from value function
def GetPolicy(V, P, R, gamma, prob="min"):
    if len(R.shape) == 2:
        R = expand(R)
    m, n, l = P.shape
    Vals = [
        [sum(P[a, i, :] * (R[a, i, :] + gamma * V)) for a in range(m)] for i in range(n)
    ]
    if prob == "min":
        pi = [np.argmin(Vals[i]) for i in range(n)]
    if prob == "max":
        pi = [np.argmax(Vals[i]) for i in range(n)]
    return pi


#################################################################


#################################################################
def SolveMDP(
    P,
    R,
    gamma=0.99,
    epsilon=10 ** (-10),
    p=False,
    prob="min",
    method="Value",
    threshold=float("inf"),
):
    # P: Transition probability
    # R: Reward matrix
    # epsilon: convergence param of value iteration
    # prob: Specify if the objective is to minimize ('min') or maxmize ('max') outcome
    m, n, n = P.shape
    if len(R.shape) < 3:
        R = expand(R, n, m)
    if method == "Value":
        V = np.array(ValueIteration(P, R, gamma, epsilon, prob, threshold))
        pi = np.array(GetPolicy(V, P, R, gamma, prob))

    if p:
        print("Optimal Value:", V)
        print("Optimal Policy:", pi)
    return (V, pi)


# Bellman operator for reduced R
def T_r(V, P, R, gamma, prob="min", reduced=True):
    m, n, n = P.shape
    res = np.zeros(n)
    v = V.copy()
    if prob == "min":
        for i in range(n):
            res[i] = R[i] + sum(P[0, i, :] * (gamma * v))
            for a in range(m):
                res[i] = min(res[i], R[i] + sum(P[a, i, :] * (gamma * v)))
    if prob == "max":
        for i in range(n):
            for a in range(m):
                res[i] = max(res[i], R[i] + sum(P[a, i, :] * (gamma * v)))
    return res


def fast_solve_MDP(P, R, V_start, gamma=0.9, prob="min", epsilon=10 ** (-10)):
    m, n, n = P.shape
    V = V_start
    W = T_r(V, P, R, gamma, prob)
    while np.linalg.norm(V - W) > epsilon:
        V = W
        W = T_r(W, P, R, gamma, prob)
    return W


def LP_form(P, R, gamma=0.9):
    if len(R.shape) == 2:
        R = expand(R)
    n = len(P[0])
    m = len(P)
    c = np.ones(n)
    mdplp = Model("MDP LP")
    J = np.array([None for i in range(n)])
    for i in range(n):
        J[i] = mdplp.addVar(lb=-GRB.INFINITY, name="J[%i]" % i)
    mdplp.addConstrs(
        (
            J[i] <= sum([P[a, i, j] * (R[a, i, j] + gamma * J[j]) for j in range(n)])
            for a in range(m)
            for i in range(n)
        )
    )

    mdplp.update()
    mdplp.setObjective(sum([c[j] * J[j] for j in range(n)]), GRB.MAXIMIZE)
    mdplp.params.outflag = 0
    mdplp.optimize()
    # print(mdplp.getVars())
    print("Optimal Values, solved with LP form:", [J[i].X for i in range(n)])
    return [J[i].X for i in range(n)]


def DualLP_form(P, R, gamma=0.9):
    if len(R.shape) == 2:
        R = expand(R)
    # c = np.random.rand(n)
    n = len(P[0])
    m = len(P)
    c = np.ones(n)
    mdplp_dual = Model("MDP LP DUAL")
    mu = np.array([[None for a in range(m)] for i in range(n)])
    for i in range(n):
        for a in range(m):
            mu[i, a] = mdplp_dual.addVar()
    for j in range(n):
        mdplp_dual.addConstr(
            sum(mu[j, :])
            == c[j]
            + gamma
            * sum(sum(P[a, i, j] * mu[i, a] for i in range(n)) for a in range(m))
        )
    mdplp_dual.update()
    mdplp_dual.setObjective(
        sum(
            mu[i, a] * R[a, i, j] * P[a, i, j]
            for a in range(m)
            for i in range(n)
            for j in range(n)
        ),
        GRB.MINIMIZE,
    )
    # mdplp_dual.setObjective(sum(mu[i,a]*R[a,i,j]*P[a,i,j] for a in range(m) for i in range(n) for j in range(n)), GRB.MAXIMIZE)

    mdplp_dual.params.outflag = 0
    mdplp_dual.optimize()
    # print('Optimal policy, solved with Dual LP form:',[np.max([mu[i,a].X for a in range(m)]) for i in range(n) ])
    print("Optimal Values, solved with Dual LP form:", mdplp_dual.Pi)
    return mdplp_dual.Pi


# Generate Data Samples from an MDP sample[i,t] = (x^i_t,a^i_t, r)
def sample_MDP(P, R, N, T):
    samples = np.array([[None for t in range(T)] for i in range(N)])
    m, n, n = P.shape
    for i in range(N):
        s = int(n * np.random.rand())
        for t in range(T):
            a = int(m * np.random.rand())
            s_prime = choices(range(n), P[a, s, :])[0]
            samples[i, t] = (s, a, R[s])
            s = s_prime
    return samples


# Creating normal distribution generators. normal_dist(mu,sigma)() generates a random sample.
def normal_dist(mu, sigma):  # mean  # std
    return lambda: np.random.multivariate_normal(mu, sigma)


# Example of distributiom array generation. Features are of size 2 here (ie in 2D).
# distributions = [normal_dist([mu,mu],[[0.5,0],[0,0.5]]) for mu in range(n)]
#################################################################


#################################################################
# Samples an MDP and at each state s, produces features with a certain probability distribution distributions[s]
def sample_MDP_with_features(
    P,
    R,  # MDP parameters
    distributions,  # distribution[s] should be a function f, such that f() generates a radom element following this distribution
    N,
    T,  # sample size
):
    # Output is of the form samples[i,t] = (x^i_t,a^i_t,r^i_t,s) : (features vector, action taken,reward, state from which it was sampled). Last element is usuful to evelute later on the clustering.
    samples = np.array([[None for t in range(T)] for i in range(N)])
    m, n, n = P.shape
    for i in range(N):
        s = int(n * np.random.rand())
        for t in range(T):
            x = distributions[s]()
            a = int(m * np.random.rand())
            s_prime = choices(range(n), P[a, s, :])[0]
            samples[i, t] = (x, a, R[s], s)
            s = s_prime
    return samples


# Same as before but in a diffrent format, one long list
def sample_MDP_with_features_list(
    P,
    R,  # MDP parameters
    distributions,  # distribution[s] should be a function f, such that f() generates a radom element following this distribution
    N,
    T,  # sample size
):
    samples = []
    m, n, n = P.shape
    for i in range(N):
        s = int(n * np.random.rand())
        for t in range(T):
            x = distributions[s]()
            a = int(m * np.random.rand())
            s_prime = choices(range(n), P[a, s, :])[0]
            samples.append((i, t, x, a, R[s], s))
            s = s_prime
    return samples


# Samples_action[a] : samples' points that have action a : (i,t,r)
def select_action(samples, m):
    samples_action = [[] for a in range(m)]
    N, T = samples.shape
    for a in range(m):
        for i in range(N):
            for t in range(T):
                if samples[i, t][1] == a:
                    samples_action[a].append((i, t, samples[i, t][2]))
    return np.array(samples_action)


# n,m,k=6,3,3
# N,T = 6,3
# P,R = Generate_random_MDP(6,3)
# samples = sample_MDP(P,R,N,T)


# Clusters randomly an MDP
def Random_Clustering(n, k):  # number of elements to cluster  # number of clusters
    # Generating random clustering
    deltar = np.zeros((n, k))
    for i in range(n):
        l = int(k * np.random.rand())
        deltar[i, l] = 1
    return deltar


def Construct_Clustered_MDP(P, R, delta):
    m, n = R.shape
    n, k = delta.shape
    Pc = np.zeros((m, k, k))
    Rc = np.zeros((m, k))
    for a in range(m):
        for s in range(k):
            for sp in range(k):
                Pc[a, s, sp] = sum(
                    sum(P[a, i, j] * delta[i, s] * delta[j, sp] for i in range(n))
                    for j in range(n)
                ) / max(1, sum(delta[i, s] for i in range(n)))
                Rc[a, s] = sum(R[a, i] * delta[i, s] for i in range(n)) / max(
                    1, sum(delta[i, s] for i in range(n))
                )
    return Pc, Rc


def Cluster_size(delta):
    n, k = delta.shape
    return [sum(delta[:, s]) for s in range(k)]


def accuracy_clustering_policy(delta, pi):
    k = len(delta[0])
    acc = []
    for s in range(k):
        acc.append(pi[delta[:, s] == 1])
        acc[s] = [
            int(100 * sum(acc[s] == a) / max(1, len(acc[s]))) / 100 for a in range(m)
        ]
    print("Distribution of policy in the clustering:", acc)
    return acc


# n,m,k = 8,3,3
# P,R = Generate_random_MDP(n,m)
# delta = Random_Clustering(n,k)
# Pc,Rc = Construct_Clustered_MDP(P,R,delta)
# Rc.shape


# Clusters randomly a data
def Data_Random_Clustering(
    N, T, k  # number of elements to cluster
):  # number of clusters
    # Generating random clustering
    deltar = np.zeros((N, T, k), dtype=int)
    for i in range(N):
        for t in range(T):
            s = int(k * np.random.rand())
            deltar[i, t, s] = 1
    return deltar


def Data_Construct_Clustered_MDP(samples, m, delta):
    N, T, k = delta.shape
    Pc = np.zeros((m, k, k))
    Rc = np.zeros(k)
    samples_action = select_action(samples, m)
    for a in range(m):
        for s in range(k):
            for sp in range(k):
                Pc[a, s, sp] = sum(
                    delta[u[0], u[1], s] * delta[u[0], u[1] + 1, sp]
                    for u in samples_action[a]
                    if u[1] < T - 1
                ) / max(
                    1,
                    sum(
                        delta[u[0], u[1], s] for u in samples_action[a] if u[1] < T - 1
                    ),
                )
            Rc[s] = sum(
                sum(samples[i, t][2] * delta[i, t, s] for i in range(N))
                for t in range(T)
            ) / sum(sum(delta[i, t, s] for i in range(N)) for t in range(T))
    return Pc, Rc


def Data_Cluster_size(delta):
    k = delta.shape[2]
    return [sum(sum(delta[:, :, s])) for s in range(k)]


def Data_action_Cluster_size(samples, delta, m):
    N, T, k = delta.shape
    samples_action = select_action(samples, m)
    return np.array(
        [
            [
                sum(delta[u[0], u[1], s] for u in samples_action[a] if u[1] < T - 1)
                for s in range(k)
            ]
            for a in range(m)
        ]
    )


def which_cluster(delta):
    N, T, k = delta.shape
    clust = np.zeros((N, T), dtype=int)
    for i in range(N):
        for t in range(T):
            clust[i, t] = np.where(delta[i, t])[0][0]
    return clust


# When generating samples from a larger MDP, if each sample is marked by the state from which it comes, the following function clusters the samples depending on states from which they were generated.
def state_cluster_samples(samples, k):
    N, T = samples.shape
    delta = np.zeros((N, T, k), dtype=int)
    for i in range(N):
        for t in range(T):
            delta[i, t, samples[i, t][3]] = 1
    return delta


# Getting optimal policy for a given clustering
def policy_from_clustering(samples, delta):
    N, T = samples.shape
    cluster_of = which_cluster(delta)
    Pc, Rc = Data_Construct_Clustered_MDP(samples, m, delta)
    Vc, pic = SolveMDP(Pc, Rc, p=False)
    pi_data = np.array([[pic[cluster_of[i, t]] for t in range(T)] for i in range(N)])
    return Pc, Rc, pi_data


def norm2_dist(x, y):
    return np.linalg.norm(x - y)


# Infos on ideal clustering based on the ground truth MDP
def truth_based_clustering(
    samples, k, outcome_dist=norm2_dist, transition_dist=norm2_dist
):
    delta_s = state_cluster_samples(samples, k)
    cluster_of_s = which_cluster(delta_s)
    c_s = Data_Cluster_size(delta_s)
    Pc_s, Rc_s = Data_Construct_Clustered_MDP(samples, m, delta_s)
    Vc_s = fast_solve_MDP(Pc_s, Rc_s, V_start=np.ones(n))
    cluster_distance_s = cluster_dist(samples, delta_s, transition_dist)
    outcome_loss_s, transition_loss_s = (
        outcome_pred_loss(samples, delta_s, cluster_of_s, outcome_dist, Rc_s),
        transition_pred_loss(samples, delta_s, cluster_distance_s, cluster_of_s, Pc_s),
    )
    c_a_s = Data_action_Cluster_size(samples, delta_s, m)
    return (
        delta_s,
        cluster_of_s,
        Pc_s,
        Rc_s,
        Vc_s,
        outcome_loss_s,
        transition_loss_s,
        c_s,
        c_a_s,
    )


def clustering_charachteristics(
    samples, delta, outcome_dist=norm2_dist, transition_dist=norm2_dist
):
    k = delta.shape[2]
    cluster_distance = cluster_dist(samples, delta, transition_dist)
    Pc, Rc = Data_Construct_Clustered_MDP(samples, m, delta)
    cluster_of = which_cluster(delta)
    c = Data_Cluster_size(delta)
    Vc = fast_solve_MDP(Pc, Rc, V_start=np.ones(n))
    outcome_loss, transition_loss = (
        outcome_pred_loss(samples, delta, cluster_of, outcome_dist, Rc),
        transition_pred_loss(samples, delta, cluster_distance, cluster_of, Pc),
    )
    c_a = Data_action_Cluster_size(samples, delta, m)
    return (
        cluster_of,
        Pc,
        Rc,
        Vc,
        outcome_loss,
        transition_loss,
        cluster_distance,
        c,
        c_a,
    )


# Evaluates a policy on a data generated from an MDP
def evaluate_data_policy(samples, P, R, pi_data):
    m, n, n = P.shape
    N, T = samples.shape
    mu = np.zeros((n, m))
    size = np.zeros(n)
    # We construct a policy on the initial state based on what was precribed for the data. mu[s,a]=P(taking action a in state s) = percentage of data points in state s for which action a
    for i in range(N):
        for t in range(T):
            size[samples[i, t][3]] += 1
            mu[samples[i, t][3], pi_data[i, t]] += 1
    for s in range(n):
        for a in range(m):
            mu[s, a] = mu[s, a] / size[s]
    return policy_value(mu, P, R), mu


# Shows how much a stochastic policy matches a given policy. Return matching[s] = mu(s,pi(s)) probability of taking action pi(s) in s following policy mu
def policy_match(mu, pi):
    n = pi.shape[0]
    return np.array([mu[s, pi[s]] for s in range(n)])


# Returns acc[s,s'] = percentage of elements of cluster s that come from state s' in the orignial MDP
def cluster_accuracy(k1, delta, samples):
    N, T, k = delta.shape
    clust = which_cluster(delta)
    acc = np.zeros((k, k1))
    for i in range(N):
        for t in range(T):
            s = clust[i, t]
            s1 = samples[i, t][3]
            acc[s, s1] += 1
    for s in range(k):
        acc[s] = acc[s] / max(1, sum(acc[s]))
    return acc


#################################################################
