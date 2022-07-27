"""
Measures file for different metrics to be used in the processing scripts.

Katerina Capouskova 2020, kcapouskova@hotmail.com
"""
import logging
import os
import json
import bct
import networkx as nx
from itertools import permutations
from permute.core import two_sample
import numpy as np

from utilities import create_dir


def global_efficiency_weighted(G):
    """
    Computes the average weighted global efficiency of a graph
    In Dijkstra algorithm distances are calculated as sums of weighted edges
    traversed.

    :param G: undirected weighted connection matrix (all weights in W must be between 0 and 1)
    :type G: NxN np.ndarray
    :return: average global efficiency measure
    :rtype: float
    """
    ge = bct.distance.efficiency_wei(G, local=False)
    return ge


def get_node_communities(partitions):
    """
    Returns dictionary for each node [key] with its community belonging [value]
    computed as: number of nodes that occupy in t the same community with the
    node as in t-1/the total number of nodes in a community in t-1; for each
    time point

    :param partitions: community partitions
    :type partitions: list
    :return: nodes_dict, nodes_dict_int
    :rtype: dict, dict
    """
    nodes_dict = {}
    nodes_dict_int = {}
    for part_t1, part_t2 in zip(partitions[0:-1], partitions[1:]):
        for (key, value), (key2, value2) in zip(part_t1.items(), part_t2.items()):
            keys_t1 = [k for k in part_t1 if part_t1[k] == value]
            keys_t2 = [k2 for k2 in part_t2 if part_t2[k2] == value2]
            list_intersections = [x for x in keys_t1 if x in keys_t2]
            if key in nodes_dict:
                # Key exist in dict.
                # Check if type of value of key is list or not
                if not isinstance(nodes_dict[key], list):
                    # If type is not list then make it list
                    nodes_dict[key] = [nodes_dict[key]]
                    nodes_dict_int[key] = [nodes_dict_int[key]]
                # Append the value in list
                nodes_dict[key].append(str(len(list_intersections)) + '/' + str(len(keys_t1)))
                nodes_dict_int[key].append(len(list_intersections)/len(keys_t1))
            else:
                # As key is not in dict,
                # so, add key-value pair
                nodes_dict[key] = str(len(list_intersections)) + '/' + str(len(keys_t1))
                nodes_dict_int[key] = len(list_intersections) / len(
                    keys_t1)

    return nodes_dict, nodes_dict_int


def permutation_t_test(group_a, group_b, output_path):
    """
    Computes a permutation test based on a t-statistic. Returns and t value,
    p value, a H0 for two groups.

    :param group_a: clusters array of a first group
    :type group_a: np.ndarray
    :param group_b: clusters array of a second group
    :type group_b: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    """
    create_dir(output_path)
    logging.basicConfig(
        filename=os.path.join(output_path, 'permutation_t_test.log'),
        level=logging.INFO)
    p, t = two_sample(group_a, group_b, reps=5000, stat='t',
                      alternative='two-sided', seed=20)
    logging.info('Permutation T-test value: {}, p-value: {}'.format(t, p))
    dict = {'Permutation T-test value': t, 'p-value': p}
    with open(os.path.join(output_path, 'permutation_t_test.json'), 'w') as fp:
        json.dump(dict, fp)
    return p, t


def probability_of_states(clusters, n_clusters):
    """
    Computes the probability of states.

    :param clusters: clusters array
    :type clusters: np.ndarray
    :param n_clusters: number of clusters
    :type n_clusters: int
    :return: dictinory {state: probability}
    :rtype: dict
    """
    dict_p = {}
    for n in range(n_clusters):
        n_list = [c for c in clusters if c == n]
        p = float(len(n_list))/float(len(clusters))
        dict_p.update({int(n): p})

    return dict_p


def mean_lifetime_of_states(clusters, n_clusters, TR):
    """
    Computes the mean lifetime of states.

    :param clusters: clusters array
    :type clusters: np.ndarray
    :param n_clusters: number of clusters
    :type n_clusters: int
    :param TR: imaging repetition time in seconds
    :type TR: float
    :return: dictinory {state: mean lifetime}
    :rtype: dict
    """
    dict_lt = {}
    for n in range(n_clusters):
        state_true = []
        for c in clusters:
            if c == n:
                state_true.append(1)
            else:
                state_true.append(0)
        # create differences list
        diff_list = np.diff(state_true).tolist()
        # detect swithces in and out of states
        out_state = [i for i, j in enumerate(diff_list, 1) if j == 1]
        in_state = [i for i, j in enumerate(diff_list, 1) if j == -1]
        # discard cases where state starts or ends
        if len(in_state) > len(out_state):
            in_state.pop(0)
        elif len(out_state) > len(in_state):
            out_state.pop(-1)
        elif out_state and in_state and out_state[0] > in_state[0]:
            in_state.pop(0)
            out_state.pop(-1)
        else:
            pass
        if out_state and in_state:
            # minus two lists
            c_duration = []
            for i, z in zip(in_state, out_state):
                diff = i - z
                c_duration.append(diff)
        else:
            c_duration = 0
        mean_duration = np.mean(c_duration) * TR
        dict_lt.update({int(n): mean_duration})

    return dict_lt
