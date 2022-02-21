"""
Visualization tools.

Katerina Capouskova 2018-2020, kcapouskova@hotmail.com
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


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
    plt.savefig(os.path.join(output_path, 'Val_loss_autoencoder.pdf'))


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
    plot_path = os.path.join(output_path, 'modularity_ge_plot.pdf')
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
        plot_path = os.path.join(output_path, 'node_conn_barplot_{}.pdf'.format(ta))
        sns.despine()
        plt.xlim(0.5)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.tick_params(axis="x", labelsize=36)
        ax.tick_params(axis="y", labelsize=16)
        plt.savefig(plot_path, dpi=400)


def plot_clustering_scatter(X, labels, output_path):
    """
    Plots the modularity and global efficiency of dfcs histograms

    :param X: features array
    :type X: np.ndarray
    :param labels: labels array
    :type labels: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    """
    plt.style.use('seaborn-whitegrid')
    plt.scatter(X[:, 0], X[:, 1], c=labels,
                s=50, cmap='viridis')
    plt.xlabel('Modularity')
    plt.ylabel('Global Efficiency')
    plt.title('Modularity Global Efficiency Clustering')
    plt.savefig(os.path.join(output_path, 'clustering_scatter.pdf'))


def plot_silhouette_analysis(X, output_path, n_clusters, silhouette_avg,
                             sample_silhouette_values, cluster_labels, centers):
    """
    Plots the Silhouette analysis for clustering algorithm.

    :param X: clustering input features
    :type X: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :param n_clusters: number of clusters
    :type n_clusters: int
    :param silhouette_avg: silhouette average score
    :type silhouette_avg: float
    :param sample_silhouette_values: silhouette scores for each sample
    :type sample_silhouette_values: float
    :param cluster_labels: cluster labels
    :type cluster_labels: int
    :param centers: coordinates of cluster centers
    :type centers: array, [n_clusters, n_features]
    """
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = ['darkorange', 'mediumslateblue', 'mediumaquamarine', 'orchid',
                 'steelblue', 'lightgreen', 'lightslategrey', 'darksalmon',
                 'tomato', 'turquoise', 'red', 'green', 'royalblue', 'gold', 'navy', 'violet',
                 'brown', 'seagreen', 'maroon', 'darkcyan']
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color[i], edgecolor=color[i], alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="orangered", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = plt.get_cmap('Spectral')(np.linspace(0, 1, 10))
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_path, 'Silhouette.pdf'))
    # plt.show()
