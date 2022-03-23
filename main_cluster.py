"""
Script that performs clustering on modularity and gloal efficiency features
Katerina Capouskova 2022, kcapouskova@hotmail.com
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from autoencoder import cluster_kmeans
from utilities import return_paths_list, create_dir


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run cluster module.')

    parser.add_argument('--input', type=str, help='Path to the input directory',
                        required=True)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--dfcs', nargs='+', help='Path to the input directory of dfcs',
                        required=True)

    return parser.parse_args()


def main():
    """
    Clustering algorithm
    """
    args = parse_args()
    input_path = args.input
    output_path = os.path.normpath(args.output)
    dfc_path = args.dfcs

    create_dir(output_path)
    df = pd.read_csv(input_path)
    dfc_paths = []

    # Prepare all the input paths
    for dfc_p in dfc_path:
        all_paths = return_paths_list(dfc_p, '.npz')
        dfc_paths += all_paths

    # Cluster according to modularity and global efficiency features
    X = df[["modularity", "global_efficiency"]].to_numpy()
    y_kmeans, centers = cluster_kmeans(X, output_path)
    df['y_kmeans'] = y_kmeans.tolist()

    # sort dFCs according to clusters
    for dfc in tqdm(dfc_paths):
        path, file = os.path.split(dfc)
        split_list = file.split('_')
        subject = split_list[1]
        time = split_list[3].split('.')[0]
        path_task = dfc.split('/')
        task = path_task[7]
        output_path_1 = os.path.join(output_path, 'cluster_1', task)
        create_dir(output_path_1)
        output_path_2 = os.path.join(output_path, 'cluster_2', task)
        create_dir(output_path_2)
        condition = (df['subject_x'] == int(subject)) & (df['time_x'] == int(time)) & (
                    df['task_x'] == task)
        select_index = df.index[condition].tolist()[0]
        column_loc = df.columns.get_loc('y_kmeans')
        if df.iloc[select_index, column_loc] == 0:
            shutil.copyfile(dfc, os.path.join(output_path_1, file))
        else:
            shutil.copyfile(dfc, os.path.join(output_path_2, file))


if __name__ == '__main__':
    main()
