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

    def __init__(self, dataset_path, list_ids, labels, batch_size, shuffle, multimodal = False):
        """Initialization"""

        self.indexes = np.arange(len(list_ids))
        self.batch_size = batch_size
        self.labels = labels
        self.n_classes = len(list(labels.values())[0])
        self.list_ids = list_ids
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.input_number = 1 if not multimodal else self.n_inputs()
        self.multimodal = multimodal
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
        y = np.empty((self.batch_size, self.n_classes))
        if self.multimodal:
            x = [np.empty((self.batch_size, shape[1], shape[2], 3)) for k in
                 range(self.input_number)]
            for i, ID in enumerate(list_ids_temp):
                sample = np.load(self.dataset_path + '/samples/' + ID + '.npy')
                for j, data in enumerate(self.select_sensors(sample)):
                    x[j][i,] = data
                y[i] = self.labels[ID]
        else:
            x = np.empty((self.batch_size, shape[1], shape[2], shape[3]))
            for i, ID in enumerate(list_ids_temp):
                sample[i,] = np.load(self.dataset_path + '/samples/' + ID + '.npy')
                y[i] = self.labels[ID]

        return x, y

    def n_inputs(self):
        dataset_name = self.dataset_path.split("/")[-1]
        sample = np.load(self.dataset_path + '/samples/' + self.list_ids[0] + '.npy')
        input_vec = self.select_sensors(sample)

        return len(input_vec)


    def select_sensors(self, sample):
        dataset_name = os.path.normpath(self.dataset_path).split(os.path.sep)[-1]
        data = []

        if dataset_name == 'MHEALTH':
            data.append(sample[:, :, 0:3]) # ACC chest-sensor
            data.append(sample[:, :, 5:8])  # ACC left-ankle sensor
            data.append(sample[:, :, 8:11]) # GYR left-ankle sensor
            data.append(sample[:, :, 11:14])  # MAG left-ankle sensor
            data.append(sample[:, :, 14:17])  # ACC right-lower-arm
            data.append(sample[:, :, 17:20])  # GYR right-lower-arm
            data.append(sample[:, :, 20:23])  # MAG right-lower-arm

        elif dataset_name == 'PAMAP2P':
            data.append(sample[:, :, 1:4]) # ACC1 over the wrist on the dominant arm
            data.append(sample[:, :, 4:7]) # ACC2 over the wrist on the dominant arm
            data.append(sample[:, :, 7:10]) # GYR over the wrist on the dominant arm
            data.append(sample[:, :, 10:13]) # MAG over the wrist on the dominant arm
            data.append(sample[:, :, 14:17]) # ACC1 chest-sensor
            data.append(sample[:, :, 17:20]) # ACC2 chest-sensor
            data.append(sample[:, :, 20:23]) # GYR chest-sensor
            data.append(sample[:, :, 23:26]) # MAG chest-sensor
            data.append(sample[:, :, 27:30]) # ACC1 on the dominant side's ankle
            data.append(sample[:, :, 30:33]) # ACC2 on the dominant side's ankle
            data.append(sample[:, :, 33:36]) # GYR on the dominant side's ankle
            data.append(sample[:, :, 36:39]) # MAG on the dominant side's ankle

        elif dataset_name == 'UTD-MHAD1_1s' or dataset_name == 'UTD-MHAD2_1s' or dataset_name == 'USCHAD':
            # UTD-MHAD1_1s: ACC right-wrist
            # UTD-MHAD2_1s: ACC right-thigh
            # USCHAD: ACC subject’s front right hip inside a mobile phone pouch
            data.append(sample[:, :, 0:3])
            # UTD-MHAD1_1s: GYR right-wrist
            # UTD-MHAD2_1s: GYR right-thigh
            # USCHAD: GYR subject’s front right hip inside a mobile phone pouch
            data.append(sample[:, :, 3:6])

        elif dataset_name == 'WHARF' or dataset_name == 'WISDM':
            # WHARF: ACC right-wrist
            # WISDM: ACC 5 different body positions (apparently)
            data.append(sample[:, :, 0:3])

        else:
            sys.exit("Dataset name {} is wrong.".format(dataset_name))

        return data

    def get_shape(self):
        """Get dataset shape"""

        sample = np.load(self.dataset_path + '/samples/' + self.list_ids[0] + '.npy')
        if self.multimodal:
            shape = (len(self.list_ids), sample.shape[0], sample.shape[1], 3)
        else:
            shape = (len(self.list_ids), sample.shape[0], sample.shape[1], sample.shape[2])

        return shape

    def get_nclasses(self):
        """Get number of classes"""
        return self.n_classes

    def get_moda_names(self):
        dataset_name = os.path.normpath(self.dataset_path).split(os.path.sep)[-1]
        data = []

        if dataset_name == 'MHEALTH':
            names = ["a_chest", "a_left-ankle", "g_left-ankle", "m_left-ankle",
                     "a_right-wrist", "g_right-wrist", "m_right-wrist"]

        elif dataset_name == 'PAMAP2P':
            names = ["a1_dominant-wrist", "a2_dominant-wrist", "g_dominant-wrist", "m_dominant-wrist",
                     "a1_chest", "a2_chest", "g_chest", "m_chest",
                     "a1_dominant_ankle", "a2_dominant_ankle", "g_dominant_ankle", "m_dominant_ankle"]

        elif dataset_name == 'UTD-MHAD1_1s':
            names = ["a_right-wrist", "g_right-wrist"]

        elif dataset_name == 'UTD-MHAD2_1s':
            names = ["a_right-thigh", "g_right-thigh"]
        elif dataset_name == 'USCHAD':
            names = ["a_front-right-hip", "g_front-right-hip"]

        elif dataset_name == 'WHARF':
            names = ["a_right-wrist"]
        elif dataset_name == 'WISDM':
            names = ["acc"]

        else:
            sys.exit("Dataset name {} is wrong.".format(dataset_name))

        return names

