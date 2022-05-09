"""
Script that performs autoencoder on clusters of modularity and gloal efficiency features
Katerina Capouskova 2022, kcapouskova@hotmail.com
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from autoencoder import cluster_kmeans, autoencoder
from utilities import return_paths_list, create_dir, preprocess_autoencoder


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run encode module.')

    parser.add_argument('--input', nargs='+', help='Path to the input directories',
                        required=True)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--latent', type=int, default=2,
                        help='Number of dimensions in the latent space', required=False)
    parser.add_argument('--ba', type=int, default=80,
                        help='Number of brain areas',
                        required=False)
    parser.add_argument('--imb', action='store_true', default=False,
                        help='Imbalanced dataset', required=False)

    return parser.parse_args()


def main():
    """
    Autoencoder algorithm
    """
    args = parse_args()
    input_path = args.input
    output_path = os.path.normpath(args.output)
    brain_areas = args.ba
    latent = args.latent
    imbalanced = args.imb

    create_dir(output_path)
    #dfc_paths = return_paths_list(input_path, '.npz')
    dfc_all, n_samples, y = preprocess_autoencoder(input_path, output_path,
                                                   brain_areas)
    encoded = autoencoder(dfc_all, output_path, y, latent, imbalanced=imbalanced)



if __name__ == '__main__':
    main()
