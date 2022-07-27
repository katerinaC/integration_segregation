"""
Script that estimates p values

Katerina Capouskova 2018, kcapouskova@hotmail.com
"""

import argparse
import itertools
import os

import pandas as pd
from statsmodels.stats.multitest import multipletests

from measures import permutation_t_test
from utilities import create_dir, find_delimeter


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run T-test.')

    parser.add_argument('--input', type=str, help='Path to the input csv file',
                        required=True)
    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)

    return parser.parse_args()


def main():
    """
    T-test
    """
    args = parse_args()
    input_path = args.input
    output_path = args.output

    create_dir(output_path)

    delim = find_delimeter(input_path)
    df = pd.read_csv(input_path, sep=delim)
    df = df[df['cluster'] == 1.]
    df = df[df['number_of_bins'] == 12.]

    tasks = ["Emotion", "Motor", "Social", "Working_memory", "Language", "Gambling", "Rest"]

    mod_p_values = []
    mod_t_values = []
    ge_p_values = []
    ge_t_values = []
    conditions = []

    for a, b in itertools.combinations(tasks, 2):
        df_cond_a = df[df['task'] == a]
        df_cond_b = df[df['task'] == b]
        p_mod, t_mod = permutation_t_test(df_cond_a['entropy'], df_cond_b['entropy'],
                                                 os.path.join(output_path, 'entropy', a + '_' + b))
        #p_ge, t_ge = permutation_t_test(df_cond_a['lifetime'],
                                        #df_cond_b['lifetime'],
                                             #os.path.join(output_path, 'lifetime', a + '_' + b))
        mod_p_values.append(p_mod)
        mod_t_values.append(t_mod)
        #ge_p_values.append(p_ge)
        #ge_t_values.append(t_ge)
        conditions.append(a + ' ' + b)

    mod_p_adjusted = multipletests(mod_p_values, alpha=0.05, method='bonferroni')
    #ge_p_adjusted = multipletests(ge_p_values, alpha=0.05, method='bonferroni')
    p_values = pd.DataFrame({'conditions': conditions,
                            'bonferroni_ent_p': mod_p_adjusted[1].tolist(),
                            #'bonferroni_lt_p': ge_p_adjusted[1].tolist(),
                             'proba_p': mod_p_values,
                            #'lt_p': ge_p_values,
                            'proba_t': mod_t_values,
                             #'lt_t': ge_t_values
                             })
    p_values.to_csv(os.path.join(output_path, 'p_values_ent_2.csv'))


if __name__ == '__main__':
    main()
