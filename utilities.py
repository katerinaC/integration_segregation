"""
Utilities file for different operations to be used in the processing scripts.
Mainly copied from brain dynamics repo.

Katerina Capouskova 2018-2020, kcapouskova@hotmail.com
"""
import json
import os

import networkx as nx
import numpy as np
import scipy.io
from tqdm import tqdm


def return_paths_list(input_path, pattern):
    """
    Loads the .mat file and saves it as a .csv file or files. Returns all
    paths in a directory with specific format as a list.

    :param input_path: path to the files directory
    :type input_path: str
    :param pattern: the pattern of files to include
    :type pattern: str
    :return: list of all paths in a directory
    :rtype: []
    """
    # list of all .csv files in the directory
    paths_list = []
    if pattern == '.csv' or pattern == '.npz':
        for directory, _, files in os.walk(input_path):
            paths_list += [os.path.join(directory, file) for file in files
                           if file.endswith(pattern)]
    return paths_list


def trasform_data(input_path, output_path, n_subjects, n_tasks):
    """
    Loads data in npy format and outputs them in a desired format.
    For each subject get Time X Brain Areas matrix as a .csv file

    :param input_path: path to the file
    :type input_path: str
    :param output_path: path where to save the .csv file/s
    :type output_path: str
    :param n_subjects: number of subjects
    :type n_subjects: int
    :param n_tasks: number of tasks
    :type n_tasks: int
    """
    if os.path.isdir(output_path):
        pass
    else:
        os.makedirs(output_path)
    data = np.load(input_path)
    data = np.swapaxes(data, 2, 3)
    for subject in range(n_subjects):
        for task in range(n_tasks):
            np.savetxt(os.path.join(output_path, 'subject{}_task{}.csv'.format
            (subject, task)), data[subject, task, :, :], delimiter=',')


def return_empty_array_rows_columns(input_path, output_path):
    """
    Returns an empty array with number or rows and columns according to input
    files.

    :param input_path: path to the files dir
    :type input_path: str/list
    :param output_path: path where to save the file
    :type output_path: str
    :return: empty array
    :rtype: np.ndarray
    """
    if not isinstance(input_path, (list, tuple)):
        input_path = return_paths_list(input_path, output_path, pattern='.npz')
    else:
        pass
    n_rows_list = []
    n_columns_list = []
    for path in input_path:
        reduced_components = np.load(path)['arr_0']
        samples, timesteps, features = reduced_components.shape
        n_rows_list.append((samples * timesteps))
        n_columns_list.append(features)
    n_rows = sum(n_rows_list)
    n_columns = max(n_columns_list)
    array = np.full((n_rows, n_columns), fill_value=np.nan, dtype=np.float64)
    return array


def create_new_output_path(input_path, output_path):
    """
    Returns a new output_path from input_path

    :param input_path: path to the files dir
    :type input_path: str
    :param output_path: path where to save the file
    :type output_path: str
    :return: new output path
    :rtype: str
    """
    base_name = os.path.basename(input_path)
    return os.path.join(output_path, base_name)


def create_dir(output_path):
    """
    Creates a new directory for output path

    :param output_path: path to output dir
    :type output_path: str
    """
    if os.path.isdir(output_path):
        pass
    else:
        os.makedirs(output_path)


def separate_concat_array(input_path, starts_json, output_path, n_clusters):
    """
    Separate concatenated array with clustered states according to json with its
    starting points.

    :param input_path: path to the concatenated array
    :type input_path: str
    :param starts_json: path to json with starts
    :type starts_json: str
    :param output_path: path where to save the file
    :type output_path: str
    :param n_clusters: number of clusters
    :type n_clusters: int
    :return: output_paths: list of new output paths
    :rtype: list
    """
    data = np.load(input_path)['arr_0']
    starts = json.load(open(starts_json))
    output_paths = []
    for n in tqdm(range(len(starts))):
        new_array = data[starts.items()[n][1][0]:starts.items()[n][1][1], :]
        output = os.path.join(output_path, starts.keys()[n],
                              'splitted_matrix_clusters.npz')
        output_paths.append(output)
        create_dir(os.path.join(output_path, starts.keys()[n]))
        np.savez_compressed(output, new_array)
    return output_paths


def preprocess_autoencoder(input_paths, output_path, brain_areas):
    """
    Preprocesses data for autoencoder. Takes all dynamic functional connectivity
    matrices and concatenates them together. Also, creates array_starts.json for
    further processing.

    :param input_paths: paths to input directories
    :type input_paths: []
    :param output_path: path to output directory
    :type output_path: str
    :param brain_areas: number of brain areas
    :type brain_areas: int
    :return: array with all dfc matrices, number of samples, list of classes
    :rtype: np.ndarray, int, list
    """
    all_paths = []
    start = [0]
    y = []
    y_tasks = []
    subjects = []
    dict = {}

    for path in tqdm(input_paths):
        all_subjects_paths = return_paths_list(path, '.npz')
        n_subjects_times = len(all_subjects_paths)
        all_paths.extend(all_subjects_paths)
        for pa in all_subjects_paths:
            task = pa.split('/')[7]
            y_tasks.append(task)
        for sub in all_subjects_paths:
            subj = sub.split('/')[8].split('_')[1]
            subjects.append(subj)
        dict.update({all_subjects_paths[0].split('/')[6]: (
        start[input_paths.index(path)],
        start[input_paths.index(path)] + n_subjects_times)})
        start.append((n_subjects_times + start[input_paths.index(path)]))
        y += [input_paths.index(path) for i in range(n_subjects_times)]
    with open(os.path.join(output_path, 'arrays_starts.json'), 'w') as fp:
        json.dump(dict, fp)
    n_samples = len(all_paths)
    dfc_all = np.full((n_samples, brain_areas, brain_areas), fill_value=0).astype(np.float64)
    #dfc_all = np.memmap('merged.buffer', dtype=np.float64, mode='w+',
                       #shape=(n_samples, brain_areas, brain_areas))

    for p in tqdm(all_paths):
        dfc = np.load(p)['arr_0']
        dfc_all[all_paths.index(p), :, :] = dfc
    np.save(os.path.join(output_path, 'y'), y)
    np.save(os.path.join(output_path, 'y_tasks'), y_tasks)
    np.savez_compressed(os.path.join(output_path, 'dfc_all'), dfc_all)
    np.save(os.path.join(output_path, 'subjects'), subjects)
    return dfc_all, n_samples, np.asarray(y)


def find_delimeter(input_path):
    """
    Finds a .csv file delimeter

    :param input_path: path to a .csv file
    :type input_path: str
    """
    with open(input_path, 'r') as myCsvfile:
        header = myCsvfile.readline()
        if header.find(';') != -1:
            return ';'
        if header.find(',') != -1:
            return ','


def create_graph(input_path):
    """
    Creates a graph and indicates tasks and subjects

    :param input_path: path to a directory with dfc filea
    :type input_path: list
    :return: graph
    :rtype: networkx.Graphs
    """
    input = np.load(input_path)['arr_0']
    input = np.absolute(input)
    np.fill_diagonal(input, 0)
    A = np.asmatrix(input)
    G = nx.from_numpy_matrix(A)

    return G
