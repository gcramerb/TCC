import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input,Concatenate, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout, Conv2D,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, mean_squared_error
from scipy.stats import randint as sp_randint
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import sys
sys.path.insert(0, "../")

from utils import dataHandler
K.set_image_data_format('channels_first')


class MyAutoEncoder:

    def __init__(self):
        np.random.seed(12227)
        self.conv_window = (5, 5)
        self.pooling_window = (2, 1)
        self.n_filters = (32, 16, 8)
        self.n_epoches = 140
        self.batch_size = 8

    def build_encoder(self,input):
        #inp = np.expand_dims(input, axis=1)
        dims = input.shape

        if dims[2] == 250:
            merg_fil = (9, 1)
        elif dims[2] == 50:
            merge_fil = (3, 1)
        else:
            print('define dims!')
            merge_fil = (3,1)
        conv_1 = Conv2D(self.n_filters[0], self.conv_window, activation='relu', padding='same')(input)
        print("shape after first conv", K.int_shape(conv_1))
        pool_1 = MaxPooling2D(self.pooling_window, padding='same')(conv_1)
        print("shape after first pooling", K.int_shape(pool_1))
        conv_2 = Conv2D(self.n_filters[1], self.conv_window, activation='relu', padding='same')(pool_1)
        print("shape after second conv", K.int_shape(conv_2))
        pool_2 = MaxPooling2D((5, 1), padding='same')(conv_2)
        print("shape after second pooling", K.int_shape(pool_2))
        conv_3 = Conv2D(self.n_filters[2], self.conv_window, activation='relu', padding='same')(pool_2)
        print("shape after third conv", K.int_shape(conv_3))
        # steam = Conv2D(n_filters[2], merge_fil, activation='relu', padding='valid')(conv_3)
        encoded = conv_3
        return Model(input,encoded)


    def build_decoder(self,encoded):

        up_3 = UpSampling2D((5,1))(encoded)
        print("shape after upsample third pooling", K.int_shape(up_3))

        conv_neg_3 = Conv2D(self.n_filters[2], self.conv_window, activation='relu', padding='same')(up_3)
        print("shape after decode third conv", K.int_shape(conv_neg_3))

        up_2 = UpSampling2D(self.pooling_window)(conv_neg_3)
        print("shape after upsample second pooling", K.int_shape(up_2))

        conv_neg_2 = Conv2D(self.n_filters[1], self.conv_window, activation='relu', padding='same')(up_2)
        print("shape after decode second conv", K.int_shape(conv_neg_2))

        decoded = Conv2D(1, self.conv_window, activation='linear', padding='same')(conv_neg_2)
        return decoded

    def buildModel(self,n_sensors,dim):
        """
        Autoencoder do tipo Y. A principio entra acelerometro e Gyr e sai acc reconstruido.
        data:
        """
        inputs_keras = []
        for x in range(n_sensors):
            inputs_keras.append(Input(dim[1:]))

        streams_models = []
        for inp in inputs_keras:
            streams_models.append(self.build_encoder(inp).output)
        if len(streams_models) > 1:
            #encoded = concatenate(streams_models, axis=1)
            encoded =Concatenate(axis=1)(streams_models)
        else:
            encoded = streams_models[0]

        print("shape of encoded", K.int_shape(encoded))

        #encoded_ = MaxPooling2D(pooling_window, padding='valid')(encoded)
        decoded = self.build_decoder(encoded)


        print("shape after decode to input", K.int_shape(decoded))

        autoencoder = Model(inputs_keras, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        #encoder_model = Model(self.autoencoder.input, self.autoencoder.layers[6].output)
        self.autoencoder = autoencoder

    def fit(self,dataX,dataY):
        self.history = self.autoencoder.fit(dataX,dataY, epochs=self.n_epoches, batch_size=self.batch_size, shuffle=True)
        a = 1

DH = dataHandler()
DH.load_data(dataset_name = 'UTD-MHAD1_1s.npz',sensor_factor = '1.1')
DH.apply_missing(missing_factor = '0.4',missing_sensor = '1.0')



X_train = DH.dataX
X_test = DH.dataX
inputs_aux = []
for sensor in X_train:
    sensor = np.expand_dims(sensor,axis = 1)
    inputs_aux.append(sensor)

dim = inputs_aux[0].shape
inputs_keras = []


AE = MyAutoEncoder()
AE.buildModel(n_sensors = len(X_train),dim = dim)
AE.fit(inputs_aux,inputs_aux[0])


x_hat = autoencoder.predict(inputs_aux)
print(h_hat)



'''

inputs = np.concatenate(tuple(inputs_aux),axis = 1)
'''

#para mhealth o pooling window estava 5,1
#conv_window=(3, 3)
#pooling_window=(5, 1)