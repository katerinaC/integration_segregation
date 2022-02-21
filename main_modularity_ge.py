"""
Script that computes functional connectivity dynamics, performs dim. reduction
with autoencoder and performs modularity and global efficiency measures
Katerina Capouskova 2020, kcapouskova@hotmail.com
"""
import argparse
import json
import os
import numpy as np
import community
import pandas as pd

#from autoencoder import autoencoder
#from data_processing_functional_connectivity import dynamic_functional_connectivity
from tqdm import tqdm
from glob import glob

from data_processing_functional_connectivity import \
    dynamic_functional_connectivity
from measures import global_efficiency_weighted, get_node_communities
from utilities import create_new_output_path, create_dir, \
    preprocess_autoencoder, \
    return_paths_list, create_graph, find_delimeter
from visualizations import plot_histogram_mod_ge


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run encode ae module.')

    parser.add_argument('--input', nargs='+', help='Path to the input directory',
                        required=True)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--ba', type=str,
                        help='Path to brain areas files (.csv)', required=False)
    parser.add_argument('--pattern', type=str,
                        help='Pattern of the input file', required=True)
    parser.add_argument('--areas', type=int,
                        help='Number of brain areas', required=False)
    parser.add_argument('--phases', type=int,
                        help='Number of time phases', required=False)
    parser.add_argument('--tr', type=float,
                        help='TR of imaging method', required=True)
    parser.add_argument('--autoen', action='store_true', default=False,
                        help='Perform autoencoder data dimension reduction', required=False)
    parser.add_argument('--dfc', action='store_true', default=False,
                        help='Create dfc matrices first', required=False)
    parser.add_argument('--imb', action='store_true', default=False,
                        help='Imbalanced dataset', required=False)
    parser.add_argument('--mod', action='store_true', default=False,
                        help='Perform modularity measure on dfc', required=False)
    parser.add_argument('--node_comm', action='store_true', default=False,
                        help='Perform node community detection on dfc',
                        required=False)
    parser.add_argument('--ge', action='store_true', default=False,
                        help='Perform global efficiency measure on dfc',
                        required=False)
    return parser.parse_args()


def main():
    """
    Modularity, autoencoder, dfc, global efficiency, node community
    """
    args = parse_args()
    input_paths = args.input
    output_path = os.path.normpath(args.output)
    brain_areas = args.areas
    autoen = args.autoen
    imbalanced = args.imb
    brain_nodes = args.ba
    pattern = args.pattern
    TR = args.tr
    dfc = args.dfc
    modularity = args.mod
    ge = args.ge
    node_comm = args.node_comm
    create_dir(output_path)

    new_outputs = []
    dfc_paths = []
    output_paths = []
    dict = {}

    # Create dfc matrices from raw data
    if dfc:
        for input_path in input_paths:
            name = os.path.basename(input_path)
            paths_list = return_paths_list(input_path, pattern=pattern)
            n_subjects = len(paths_list)
            array = np.genfromtxt(paths_list[0], delimiter=';')
            brain_areas = array.shape[1]
            t_phases = array.shape[0]
            dict.update({name: [n_subjects, t_phases]})
            new_output = create_new_output_path(input_path, output_path)
            new_outputs.append(new_output)
            create_dir(new_output)
            output_paths.append(os.path.join(new_output, 'components_matrix.npz'))
            dfc_path = dynamic_functional_connectivity(
                paths_list, new_output, brain_areas, pattern, t_phases, n_subjects, TR)
            dfc_paths += dfc_path

    # If dfc matrices already exist in a specific folder, indicate input path into the folder
    # or folders for all tasks
    if dfc is False:
        for path_tasks in input_paths:
            dfc_p = return_paths_list(path_tasks, '.npz')
            dfc_paths += dfc_p

    # Perform autoencoder
    '''if autoen:
        dfc_all, n_samples, y = preprocess_autoencoder(dfc_paths, output_path,
                                                       brain_areas)
        encoded = autoencoder(dfc_all, output_path, y, imbalanced=imbalanced)'''

    # Perform modularity measure on dfc matrices (no thresholding)
    if modularity:
        modular = []
        tasks = []
        subjects = []
        times = []
        for dfc_path in tqdm(dfc_paths):
            task_n = dfc_path.split('/')
            tasks.append(task_n[7])
            subject_p, fil = os.path.split(dfc_path)
            subject_fil = fil.split('_')
            sub = subject_fil[1]
            subjects.append(sub)
            time_prep = subject_fil[3].split('.')
            time = time_prep[0]
            times.append(time)
            graph = create_graph(dfc_path)
            part = community.best_partition(graph)
            mod = community.modularity(part, graph)
            modular.append(mod)
        dict_graphs = {'task': tasks,
                       'modularity': modular, 'subject': subjects, 'time': times}
        df_mod = pd.DataFrame.from_dict(data=dict_graphs)
        df_mod.to_csv(os.path.join(output_path, 'graph_analysis_modularity.csv'))

    if node_comm:
        brain_a_names = pd.read_csv(brain_nodes, sep=(find_delimeter(brain_nodes)))
        tasks_list = []
        subjects_list = []
        nodes = []
        # get brain areas names
        if brain_nodes is not None:
            names_areas = [name for name in brain_a_names.Rois]
        else:
            names_areas = [j for j in range(brain_areas)]
        if not modularity:
            subfolders = glob(input_paths[0] + '/*/*/')
            for direct in tqdm(subfolders):
                partitions = []
                normalized_path = os.path.normpath(direct)
                path_components = normalized_path.split(os.sep)
                subject = path_components[-1]
                subjects_list.append(subject)
                task = path_components[-2]
                tasks_list.append(task)
                subject_paths = return_paths_list(direct, '.npz')
                for time_point in tqdm(subject_paths):
                    graph = create_graph(time_point)
                    part = community.best_partition(graph)
                    partitions.append(part)
                nodes_dict, nodes_dict_int = get_node_communities(partitions)
                # save into json file
                if brain_nodes is not None:
                    nodes_dict = {i : v for i, v in zip(brain_a_names.Rois, nodes_dict.values())}
                    nodes_dict_int = {i: m for i, m in zip(brain_a_names.Rois, nodes_dict_int.values())}
                with open(os.path.join(output_path, 'nodes_dictionary_{}_{}.json'.format(task, subject)), 'w') as fp:
                    json.dump(nodes_dict, fp)
                nodes.append(nodes_dict_int)
        # create empty dataframe to fill in with nodes community values
        df_comm = pd.DataFrame(columns=names_areas)
        df_comm = df_comm.append(nodes)
        # create dataframe with tasks and subjects
        df_subj = pd.DataFrame({'task': tasks_list, 'subject': subjects_list})
        # concat the two dataframes
        df_final = pd.concat([df_subj, df_comm], axis=1)
        df_final.to_csv(os.path.join(output_path, 'nodes_communities.csv'))

    # Perform global efficiency on dfc matrices (weighted)
    if ge:
        glob_eff = []
        tasks_names = []
        subjects_eff = []
        times_eff = []
        for matrix_path in tqdm(dfc_paths):
            task_n = matrix_path.split('/')
            tasks_names.append(task_n[7])
            subject_p, fil = os.path.split(matrix_path)
            subject_fil = fil.split('_')
            sub = subject_fil[1]
            subjects_eff.append(sub)
            time_pre = subject_fil[3].split('.')
            time_e = time_pre[0]
            times_eff.append(time_e)
            np_array = np.load(matrix_path)['arr_0']
            np_array = np.absolute(np_array)
            np.fill_diagonal(np_array, 0)
            glob_effic = global_efficiency_weighted(np_array)
            glob_eff.append(glob_effic)
        dict_graph = {'task': tasks_names,
                      'global_efficiency': glob_eff, 'subject': subjects_eff,
                      'time': times_eff}
        df_ge = pd.DataFrame.from_dict(data=dict_graph)
        df_ge.to_csv(os.path.join(output_path, 'graph_analysis_global_efficiency.csv'))

    # Visualize modularity and global efficiency
    if modularity and ge:
        merged_df = pd.merge(df_mod, df_ge, left_index=True, right_index=True)
        merged_df.to_csv(
            os.path.join(output_path, 'merged_df.csv'))
        plot_histogram_mod_ge(merged_df, output_path)


if __name__ == '__main__':
    main()
