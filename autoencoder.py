"""
Script for autoencoder model.
Autoencoder implemented in Keras for features dimension reduction. If using,
check for different parameters.

Katerina Capouskova 2020, kcapouskova@hotmail.com
"""
import logging
import os
import pickle

import numpy as np
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import l1, l2

from utilities import create_dir
from visualizations import plot_val_los_autoe, plot_clustering_scatter, \
    plot_silhouette_analysis, plot_acc_autoe


def autoencoder(dfc_all, output_path, y, latent, imbalanced):
    """
    Performs an autoencoder implemented in Keras framework

    :param dfc_all: array with all dfc matrices
    :type dfc_all: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    :param y: class assignment
    :type y: []
    :param imbalanced: imbalanced dataset
    :type imbalanced: bool
    :param latent: latent space dimensionality
    :type latent: int
    :return: reduced dim. array
    :rtype: np.ndarray
    """
    logging.basicConfig(filename=os.path.join(output_path,
                                              'autoencoder.log'),
                        level=logging.INFO)
    # reshape input
    all_samples, all_ft_1, all_ft_2 = dfc_all.shape
    dfc_all_2d = dfc_all.reshape(all_samples, (all_ft_1 * all_ft_2))

    # balance dataset
    if imbalanced:
        pass
        #rus = RandomOverSampler(random_state=0, replacement=True)
        #x_resampled, y_resampled = rus.fit_resample(dfc_all_2d, y)

        # train and test partition
        #x_train_o, x_test_o, y_train, y_test = train_test_split(
            #x_resampled, y_resampled, test_size=0.20, random_state=42)
        # normalize
        #normalizer = preprocessing.Normalizer().fit(x_train_o)
        #x_train = normalizer.transform(x_train_o)
        #x_test = normalizer.transform(x_test_o)
        #predict_data = normalizer.transform(dfc_all_2d)
    else:
        # train and test partition
        #x_train_o, x_test_o, y_train, y_test = train_test_split(
            #dfc_all_2d, y, test_size=0.20, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(
            dfc_all_2d, y, test_size=0.20, random_state=42)

        # scale
        '''
        standard_scaler = preprocessing.StandardScaler().fit(x_train_o)
        x_train = standard_scaler.transform(x_train_o)
        x_test = standard_scaler.transform(x_test_o)
        predict_data = standard_scaler.transform(dfc_all_2d)'''

    # PCA
    '''
    mu = x_train.mean(axis=0)
    U, s, V = np.linalg.svd(x_train - mu, full_matrices=False)
    Zpca = np.dot(x_train - mu, V.transpose())
    Rpca = np.dot(Zpca[:, :2], V[:2, :]) + mu
    err = np.sum((x_train - Rpca) ** 2) / Rpca.shape[0] / Rpca.shape[1]
    logging.info('PCA reconstruction error with 2 PCs: ' + str(round(err, 5)))'''

    # Autoencoder
    m = Sequential()
    m.add(Dense(2000, activation='relu', input_shape=((all_ft_1 * all_ft_2),))) #activity_regularizer=l2(0.001)
    m.add(Dense(500, activation='relu'))
    m.add(Dense(250, activation='relu'))
    m.add(Dense(125, activation='relu'))
    m.add(Dense(latent, activation='relu', name="bottleneck"))
    m.add(Dense(125, activation='relu'))
    m.add(Dense(250, activation='relu'))
    m.add(Dense(500, activation='relu'))
    m.add(Dense(2000, activation='relu'))
    m.add(Dense((all_ft_1 * all_ft_2), activation='sigmoid'))
    m.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    history = m.fit(x_train, x_train, batch_size=100, epochs=23, verbose=1,
                    validation_data=(x_test, x_test)) #callbacks=[es]

    encoder = Model(m.input, m.get_layer('bottleneck').output)
    Zenc = encoder.predict(dfc_all_2d)  # bottleneck representation
    np.savez_compressed(os.path.join(output_path, 'encoder_{}_features'.format(latent)), Zenc)
    Renc = m.predict(dfc_all_2d)  # reconstruction
    np.savez_compressed(os.path.join(output_path, 'autoencoder_reconstruction'), Renc)
    logging.info('MSE:{}, Val loss:{}'.format(history.history['loss'],
                                              history.history['val_loss']))
    plot_val_los_autoe(history.history['val_loss'], history.history['loss'],
                       output_path)
    plot_acc_autoe(history.history['val_accuracy'], history.history['accuracy'],
                   output_path)
    encoder.save(os.path.join(output_path,'autoencoder_model.h5'))
    # serialize model to JSON
    model_json = encoder.to_json()
    with open(os.path.join(output_path, 'autoencoder_architecture.json'), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    encoder.save_weights(os.path.join(output_path, 'autoencoder_weights.h5'))

    return Zenc


def cluster_kmeans(X, output_path):
    """
    Performs a K-means clustering algorithm on the data.

    :param X: array with all features for clustering
    :type X: np.ndarray
    :param output_path: path to output directory
    :type output_path: str
    """
    # intialize the model
    #kmeans = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                #assign_labels='kmeans')
    kmeans = KMeans(n_clusters=2, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    # save the model
    pickle.dump(kmeans, open(os.path.join(output_path,"kmeans_model.pkl"), "wb"))
    # plot the datapoints and clusters
    plot_clustering_scatter(X, y_kmeans, output_path)
    # Silhouette scores and plots
    sample_silhouette_values = silhouette_samples(X, y_kmeans)
    # Save the silhouette values
    np.save(os.path.join(output_path, 'silhouette.npy'), sample_silhouette_values)
    silhouette_avg = silhouette_score(X, y_kmeans, sample_size=500)
    plot_silhouette_analysis(X, output_path, 2, silhouette_avg,
                             sample_silhouette_values, y_kmeans, centers)

    return y_kmeans, centers
