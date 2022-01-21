"""
Measures file for different metrics to be used in the processing scripts.

Katerina Capouskova 2020, kcapouskova@hotmail.com
"""
import bct
import networkx as nx
from itertools import permutations


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
