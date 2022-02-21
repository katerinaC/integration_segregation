"""
Script that performs clustering on modularity and gloal efficiency features
Katerina Capouskova 2022, kcapouskova@hotmail.com
"""
import argparse
import os
import shutil

import pandas as pd

from autoencoder import cluster_kmeans
from utilities import return_paths_list


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run encode ae module.')

    parser.add_argument('--input', type=str, help='Path to the input directory',
                        required=True)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--dfcs', type=str, help='Path to the input directory of dfcs',
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

    df = pd.read_csv(input_path)
    dfc_paths = return_paths_list(dfc_path, '.npz')
    X = df[["modularity", "global_efficiency"]].to_numpy()
    y_kmeans, centers = cluster_kmeans(X, output_path)
    df['y_kmeans'] = y_kmeans

    # sort dFCs according to clusters
    output_path_1 = os.path.join(output_path, 'cluster_1')
    output_path_2 = os.path.join(output_path, 'cluster_2')
    for dfc in dfc_paths:
        path, file = os.path.split(dfc)
        split_list = file.split('_')
        subject = split_list[1]
        time = split_list[3].split('.')[0]
        for i in range(0, len(df)):
            if df.iloc[i]['subject_x'] == subject and df.iloc[i]['time_x'] == time:
                if df.iloc[i]['y_kmeans'] == 0:
                    shutil.move(dfc, os.path.join(output_path_1, file))
                else:
                    shutil.move(dfc, os.path.join(output_path_2, file))
            else:
                continue


if __name__ == '__main__':
    main()
