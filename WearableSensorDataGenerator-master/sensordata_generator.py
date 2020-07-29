# ---------------------------------------------------------------------------------
# Keras DataGenerator for datasets from the benchmark "Human Activity Recognition
# Based on Wearable Sensor Data: A Standardization of the State-of-the-Art"
#
# The data used here is created by the npz_to_fold.py file.
#
# (C) 2020 Jéssica Sena, Brazil
# Released under GNU Public License (GPL)
# email jessicasenasouza@gmail.com
# ---------------------------------------------------------------------------------
import numpy as np
import keras
import sys


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, dataset_path, list_ids, labels, batch_size, shuffle):
        """Initialization"""

        self.indexes = np.arange(len(list_ids))
        self.batch_size = batch_size
        self.labels = labels
        self.n_classes = len(list(labels.values())[0])
        self.list_ids = list_ids
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""

        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indexes]
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""

        shape = self.get_shape()
        x = np.empty((self.batch_size, shape[1], shape[2], shape[3]))
        y = np.empty((self.batch_size, self.n_classes))

        for i, ID in enumerate(list_ids_temp):
            x[i,] = np.load(self.dataset_path + '/samples/' + ID + '.npy')
            y[i] = self.labels[ID]

        return x, y

    def get_shape(self):
        """Get dataset shape"""

        sample = np.load(self.dataset_path + '/samples/' + self.list_ids[0] + '.npy')
        shape = (len(self.list_ids), sample.shape[0], sample.shape[1], sample.shape[2])

        return shape

    def get_nclasses(self):
        """Get number of classes"""
        return self.n_classes
