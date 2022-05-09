"""
Script that calculates entropy for the discretized feature space
Katerina Capouskova 2022, kcapouskova@hotmail.com
"""
import argparse
import os
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from tqdm import tqdm

from utilities import create_dir


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run encode entropy module.')

    parser.add_argument('--features', type=str, help='Path to the input directory'
                        'with feature npz', required=True)
    parser.add_argument('--subjects', type=str, required=True,
                        help='Path to the input directory with subjects.npy')
    parser.add_argument('--tasks', type=str, help='Path to the input directory'
                        'with y_tasks.npy', required=True)
    parser.add_argument('--clusters', type=str, help='Path to the input directory'
                        'with y.npy', required=True)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--latent', type=int, default=2,
                        help='Number of dimensions in the latent space', required=False)
    parser.add_argument('--ba', type=int, default=80,
                        help='Number of brain areas',
                        required=False)

    return parser.parse_args()


def main():
    """
    Autoencoder algorithm
    """
    args = parse_args()
    features = args.features
    output_path = os.path.normpath(args.output)
    brain_areas = args.ba
    subjects = args.subjects
    tasks = args.tasks
    clusters = args.clusters

    create_dir(output_path)

    #max number of bins to run
    bins = 20
    bins_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    #load features array
    features = np.load(features)['arr_0']
    # calculate x_min, x_max, y_min, y_max
    x = features[:, 0]
    y = features[:, 1]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    #load subjects, tasks, and clusters
    subjects = np.load(subjects)
    tasks = np.load(tasks)
    clusters = np.load(clusters)

    # create csv
    df = pd.DataFrame({'feature_1': x, 'feature_2': y})
    df['subject'] = pd.Series(subjects)
    df['task'] = pd.Series(tasks)
    df['cluster'] = pd.Series(clusters)

    # subjects, tasks, clusters
    subjects_num = df['subject'].unique().tolist()
    all_tasks = df['task'].unique().tolist()
    all_clusters = df['cluster'].unique().tolist()

    # initialize .csv for binned statistics
    df_binned = pd.DataFrame()
    for clust in all_clusters:
        df_clust = df[df['cluster'] == clust]
        for t in all_tasks:
            df_task = df_clust[df_clust['task'] == t]
            for sub in tqdm(subjects_num):
                df_sub = df_task[df_task['subject'] == sub]
                if not df_sub.empty:
                    # create binned space
                    array = df_sub[['feature_1', 'feature_2']].to_numpy()
                    arr_x = array[:, 0]
                    arr_y = array[:, 1]
                    num_of_points = arr_x.size
                    # create bins
                    for i in bins_list:
                        binned = stats.binned_statistic_2d(
                            arr_x, arr_y, None, 'count', bins=i,
                            range=[[x_min, x_max], [y_min, y_max]])
                        binned_list = binned.statistic.tolist()
                        flat_list = [item for sublist in binned_list for item in sublist]
                        divided = [z/num_of_points if z>0 else 0 for z in flat_list]
                        entropy = scipy.stats.entropy(divided)
                        entropy = round(entropy, 4)
                        df_binned = df_binned.append({'number_of_bins': i, 'subject': sub,
                                                      'cluster': clust, 'task':t,
                                                      'entropy': entropy}, ignore_index=True)
                else:
                    continue
    df_binned.to_csv(os.path.join(output_path, 'df_binned.csv'))


if __name__ == '__main__':
    main()
