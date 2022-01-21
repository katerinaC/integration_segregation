"""
Script for autoencoder model.
Autoencoder implemented in Keras for features dimension reduction. If using,
check for different parameters.

Katerina Capouskova 2020, kcapouskova@hotmail.com
"""
import logging
import os

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from tensorflow.python.layers.core import Dense

from visualizations import plot_val_los_autoe


def autoencoder(dfc_all, output_path, y, imbalanced):
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
        rus = RandomUnderSampler(random_state=0, replacement=True)
        x_resampled, y_resampled = rus.fit_resample(dfc_all_2d, y)

        # train and test partition
        x_train_o, x_test_o = train_test_split(x_resampled, test_size=0.10)
        # normalize
        normalizer = preprocessing.Normalizer().fit(x_train_o)
        x_train = normalizer.transform(x_train_o)
        x_test = normalizer.transform(x_test_o)
        predict_data = normalizer.transform(dfc_all_2d)
    else:
        # train and test partition
        x_train_o, x_test_o = train_test_split(dfc_all_2d, test_size=0.10)
        # normalize
        normalizer = preprocessing.Normalizer().fit(x_train_o)
        x_train = normalizer.transform(x_train_o)
        x_test = normalizer.transform(x_test_o)
        predict_data = normalizer.transform(dfc_all_2d)

    # PCA
    mu = x_train.mean(axis=0)
    U, s, V = np.linalg.svd(x_train - mu, full_matrices=False)
    Zpca = np.dot(x_train - mu, V.transpose())
    Rpca = np.dot(Zpca[:, :2], V[:2, :]) + mu
    err = np.sum((x_train - Rpca) ** 2) / Rpca.shape[0] / Rpca.shape[1]
    logging.info('PCA reconstruction error with 2 PCs: ' + str(round(err, 3)))

    # Autoencoder
    m = Sequential()
    m.add(Dense(2000, activation='relu', input_shape=((all_ft_1 * all_ft_2),)))
    m.add(Dense(500, activation='relu'))
    m.add(Dense(250, activation='relu'))
    m.add(Dense(125, activation='relu'))
    m.add(Dense(2, activation='linear', name="bottleneck"))
    m.add(Dense(125, activation='relu'))
    m.add(Dense(250, activation='relu'))
    m.add(Dense(500, activation='relu'))
    m.add(Dense(2000, activation='relu'))
    m.add(Dense((all_ft_1 * all_ft_2), activation='sigmoid'))
    m.compile(loss='mean_squared_error', optimizer=Adam())
    history = m.fit(x_train, x_train, batch_size=100, epochs=10, verbose=1,
                    validation_data=(x_test, x_test))

    encoder = Model(m.input, m.get_layer('bottleneck').output)
    Zenc = encoder.predict(predict_data)  # bottleneck representation
    np.savez_compressed(os.path.join(output_path, 'encoder_{}_features'.format(all_ft_1)), Zenc)
    Renc = m.predict(predict_data)  # reconstruction
    #np.savez_compressed(os.path.join(output_path, 'autoencoder_reconstruction'), Renc)
    logging.info('MSE:{}, Val loss:{}'.format(history.history['loss'],
                                              history.history['val_loss']))
    plot_val_los_autoe(history.history['val_loss'], history.history['loss'],
                       output_path)
    encoder.save(os.path.join(output_path,'autoencoder_model.h5'))
    # serialize model to JSON
    model_json = encoder.to_json()
    with open(os.path.join(output_path, 'autoencoder_architecture.json'), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    encoder.save_weights(os.path.join(output_path, 'autoencoder_weights.h5'))

    return Zenc

