"""
Script that postprocesses csv output file of node connectivity algo from
main_modularity_ge.py
Katerina Capouskova 2021, kcapouskova@hotmail.com
"""
import argparse
import os

from utilities import find_delimeter
import pandas as pd

from visualizations import create_barplot


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser('Run postprocess node conn module.')

    parser.add_argument('--output', type=str,
                        help='Path to output folder', required=True)
    parser.add_argument('--csv', type=str,
                        help='Path to node conn file (.csv)', required=True)
    parser.add_argument('--areas', type=int,
                        help='Number of brain areas', required=False)
    parser.add_argument('--tr', type=float,
                        help='TR of imaging method', required=False)
    return parser.parse_args()


def main():
    """
    Node connectivity postprocessing
    """
    args = parse_args()
    output_path = os.path.normpath(args.output)
    brain_areas = args.areas
    csv_file = args.csv
    TR = args.tr

    delimeter = find_delimeter(csv_file)
    df = pd.read_csv(csv_file, sep=delimeter)

    # drop redundant column
    df.drop(columns=['Unnamed: 0'], inplace=True)

    ba_columns = df.columns.tolist()
    ba_columns.remove('task')
    ba_columns.remove('subject')

    # get mean for each subject in a node
    for col in ba_columns:
        df[col] = df[col].apply(eval)
        df[col] = df[col].apply(lambda x: sum(x) / len(x))

    #df = df.groupby('task')[ba_columns].mean()

    df.to_csv(os.path.join(output_path, 'mean_node_connectivity_wide.csv'))

    df_long = pd.melt(df, id_vars=['task', 'subject'], value_vars=brain_areas)
    df_long.rename(columns={'variable': 'brain_area', 'value': 'node_conn'},
                   inplace=True)

    create_barplot(df_long, output_path)

if __name__ == '__main__':
    main()
