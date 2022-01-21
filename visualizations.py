"""
Visualization tools.

Katerina Capouskova 2018-2020, kcapouskova@hotmail.com
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_val_los_autoe(val, loss, output_path):
    """
    Plots the training and validation loss function of an autoencoder.

    :param val: validation loss values
    :type val: []
    :param loss: training loss values
    :type loss: []
    :param output_path: path to output directory
    :type output_path: str
    """
    sns.set(style="whitegrid")
    dict = {'validation': val, 'training': loss}
    data = pd.DataFrame(data=dict)
    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.savefig(os.path.join(output_path, 'Val_loss_autoencoder.png'))


def plot_histogram_mod_ge(df, output_path):
    """
    Plots the modularity and global efficiency of dfcs histograms

    :param df: DataFrame
    :type df: pd.DataFrame
    :param output_path: path to output directory
    :type output_path: str
    """
    sns.set(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2)
    df.rename(columns={'task_x': 'Task'}, inplace=True)
    sns.kdeplot(data=df, x="modularity", hue="Task", ax=ax1, fill=True, common_norm=False, palette="crest",
    alpha=.5, linewidth=0)
    sns.kdeplot(data=df, x="global_efficiency", hue="Task", ax=ax2, fill=True, common_norm=False, palette="crest",
    alpha=.5, linewidth=0)
    #fig.suptitle('Modularity and Global Efficiency Kernel Estimation Density')
    plot_path = os.path.join(output_path, 'modularity_ge_plot.png')
    plt.savefig(plot_path)


def create_barplot(df, output_path):
    """
    Plots the barplot for all node connectivities

    :param df: DataFrame
    :type df: pd.DataFrame
    :param output_path: path to output directory
    :type output_path: str
    """
    df = df.groupby(['task', 'brain_area'], as_index=False).mean()
    tasks = df.task.unique()
    for ta in tasks:
        df_task = df[df['task'] == ta]
        sns.set(style="whitegrid")
        plt.figure(figsize=(25, 20))
        col = []
        for val in df_task['node_conn']:
            if val <= df_task['node_conn'].nsmallest(3).iloc[2]:
                col.append('red')
            elif val >= df_task['node_conn'].nlargest(3).iloc[2]:
                col.append('gold')
            else:
                col.append('grey')
        ax = sns.barplot(x="node_conn", y='brain_area', data=df_task, palette=col)
        plot_path = os.path.join(output_path, 'node_conn_barplot_{}.png'.format(ta))
        sns.despine()
        plt.xlim(0.5)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.tick_params(axis="x", labelsize=36)
        ax.tick_params(axis="y", labelsize=16)
        plt.savefig(plot_path, dpi=400)
