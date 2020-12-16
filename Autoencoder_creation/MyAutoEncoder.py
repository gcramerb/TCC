import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Conv2D, MaxPooling2D, UpSampling2D,BatchNormalization
from keras.models import Model, load_model
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from keras import backend as K

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


K.set_image_data_format('channels_first')

from utils import dataHandler

"AutoEncoder baseado no modelo Sena2018_HAR"

# batch normalization
class Conv1DAutoEncoder():

    def __init__(self):
        np.random.seed(12227)
        self.batch_size = 128
        self.n_ep = 200

    def _stream_decode(self, encoded, n_filters, kernel):

        hidden = Conv2D(filters=n_filters[0], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
                        padding='same')(encoded)
        hidden = UpSampling2D(size=(2, 1))(hidden)
        hidden = Conv2D(filters=n_filters[1], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
                        padding='same')(hidden)
        hidden = UpSampling2D(size=(2, 1))(hidden)

        out = hidden

        return out

    def _stream(self, inp, n_filters, kernel):

        hidden = Conv2D(filters=n_filters[0], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
                        padding='same')(inp)
        hidden = MaxPooling2D(pool_size=(2, 1))(hidden)
        hidden = Conv2D(filters=n_filters[1], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
                        padding='same')(hidden)
        hidden = MaxPooling2D(pool_size=(2, 1), padding='same')(hidden)

        out = hidden

        return out


    def build_model(self,inputs):
        pool = [(2, 2), (3, 3), (5, 2), (12, 2), (25, 2)]
        pool_decode = [(25, 2), (12, 3), (5, 2), (3, 3), (2, 2)]
        width = (16, 32)
        width_decode = (32, 16)

        inputs = BatchNormalization()(inputs)


        streams_models = []
        for inp in inputs:
            for i in range(len(pool)):
                streams_models.append(self._stream(inp, width, pool[i]))

        if len(pool) > 1:
            concat = keras.layers.concatenate(streams_models, axis=1)
        else:
            concat = streams_models[0]

        # hidden = activation_layer('selu', 0.1, concat)
        encoded  = concat



        streams_models = []
        for inp in inputs:
            for i in range(len(pool)):
                streams_models.append(self._stream_decode(encoded, width_decode, pool_decode[i]))

        if len(pool) > 1:
            concat = keras.layers.concatenate(streams_models, axis=1)
        else:
            concat = streams_models[0]

        # hidden = activation_layer('selu', 0.1, concat)
        out = Conv2D(filters=1, kernel_size=(3, 1), activation='relu', kernel_initializer='glorot_normal',
                     padding='valid')(concat)

        # -------------- model buider  --------------------------------------------
        self.model = keras.models.Model(inputs=inputs, outputs=out)
        opt = keras.optimizers.Adam(learning_rate=0.005)
        self.model.compile(loss='mse',  optimizer=opt, metrics=['mse'])

    def train(self,trainX,trainY):

        inputs = []
        inputs.append(keras.layers.Input((trainX.shape[1], trainX.shape[2],trainX.shape[3])))
        self.build_model(inputs)

        self.model.fit(trainX, trainX, self.batch_size, self.n_ep, verbose=1)


    def predict(self,testX):

        x_hat = self.model.predict(testX)

        return x_hat

    def eval(self,plot = True):

        train_loss = history.history['loss']
        if plot:
            # summarize history for loss
            plt.plot(train_loss)
            #plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()



AE = Conv1DAutoEncoder()
DH = dataHandler()
a= 1
