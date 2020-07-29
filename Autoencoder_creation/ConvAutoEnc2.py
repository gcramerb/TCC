import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Conv2D, MaxPooling2D, UpSampling2D
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

from utils import get_data ,plot_sensor
import custom_model as cm

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


K.set_image_data_format('channels_last')
#if __name__ == '__main__':
np.random.seed(12227)
pool = [(2, 2), (3, 3), (5, 2), (12, 2), (25, 2)]
pool_decode = [(25, 2), (12, 3), (5, 2), (3, 3), (2, 2)]
def _stream_decode(encoded, n_filters, kernel):

    hidden = Conv2D(filters=n_filters[0], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal', padding='same')(encoded)
    hidden = UpSampling2D(size=(2,1))(hidden)
    hidden = Conv2D(filters=n_filters[1], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal', padding='same')(hidden)
    hidden = UpSampling2D(size=(2,1))(hidden)

    out = hidden

    return out


def _stream(inp, n_filters, kernel):

    hidden = Conv2D(filters=n_filters[0], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal', padding='same')(inp)
    hidden = MaxPooling2D(pool_size=(2,1))(hidden)
    hidden = Conv2D(filters=n_filters[1], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal', padding='same')(hidden)
    hidden = MaxPooling2D(pool_size=(2,1),padding='same')(hidden)

    out = hidden

    return out



def SenaBased_Eecoder(inputs, pool):
    width = (16, 32)

    streams_models = []
    for inp in inputs:
        for i in range(len(pool)):
            streams_models.append(_stream(inp, width, pool[i]))

    if len(pool) > 1:
        concat = keras.layers.concatenate(streams_models, axis=-1)
    else:
        concat = streams_models[0]

    #hidden = activation_layer('selu', 0.1, concat)
    out = concat


    return out


def SenaBased_Decoder(encoded, pool):
    width_decode = (32, 16)

    streams_models = []
    for inp in inputs:
        for i in range(len(pool)):
            streams_models.append(_stream_decode(encoded, width_decode, pool_decode[i]))

    if len(pool) > 1:
        concat = keras.layers.concatenate(streams_models, axis=-1)
    else:
        concat = streams_models[0]


    #hidden = activation_layer('selu', 0.1, concat)
    out = Conv2D(filters=1, kernel_size=(3,1), activation='relu', kernel_initializer='glorot_normal', padding='valid')(concat)

    # -------------- model buider  --------------------------------------------
    model = keras.models.Model(inputs=inputs, outputs=out)
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='mse',
                  optimizer=opt,
                  metrics=['mse'])

    return model





def get_model(shape_train):
    input_img = Input(shape=(shape_train[1], shape_train[2]))  # adapt this if using `channels_first` image data format

    x1 = Conv1D(50, (5), activation='relu', padding='same')(input_img)
    x2 = MaxPooling1D((2), padding='same')(x1)
    x3 = Conv1D(25, (5), activation='relu', padding='same')(x2)

    encoded = MaxPooling1D((1), padding='same')(x3)

    x4 = Conv1D(25, (5), activation='relu', padding='same')(encoded)
    x5 = UpSampling1D((2))(x4)
    x6 = Conv1D(50, (5), activation='relu', padding='same')(x5)
    x7 = Conv1D(3, (5), activation='relu', padding='same')(x6)
    decoded = Conv1D(3, (2), activation='sigmoid', padding='same')(x7)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return autoencoder




X,y,folds,label_names = get_data('MHEALTH')
X = np.squeeze(X)
n_class = y.shape[1]
	# for i in range(0, len(folds)):
i = 0
train_idx = folds[i][0]
test_idx = folds[i][1]
X_train = []
X_test = []

X_train = X[train_idx]
X_test = X[test_idx]
Y_train = [np.argmax(y) for y in y[train_idx]]
Y_test = [np.argmax(y) for y in y[test_idx]]

inputs = []
X_train = np.expand_dims(X_train, axis = 3)
inputs.append(keras.layers.Input((X_train.shape[1], X_train.shape[2], X_train.shape[3])))
#inputs.append(keras.layers.Input((X_train.shape[1], X_train.shape[2])))
pool = [(2, 2), (3, 3), (5, 2), (12, 2), (25, 2)]

#--------------------------------------------------------------------------------------------------------------------

#autoencoder = get_model_Sena_kernel(X_train.shape)
#autoencoder.fit([X_train], [X_train], epochs=50, batch_size=32,shuffle=True, validation_data=([X_train], [X_train]))
#encoded = SenaBased_Eecoder(inputs,pool)
#autoencoder = SenaBased_Decoder(encoded, pool_decode )
batch = 128
#autoencoder.fit(X_train, X_train, batch, cm.n_ep, verbose=0)




model_name = "ConvAutoe_Sena.h5"
#autoencoder.save(model_name)
autoencoder= load_model(model_name)

X_test = np.expand_dims(X_test, axis = 3)
X_hat = autoencoder.predict(X_test)
X_test = np.squeeze(X_test)
X_hat = np.squeeze(X_hat)
a = 0
for i in range(10,100,10):
    label = label_names[Y_test[i]]
    plot_sensor(X_test[i,:,:],X_hat[i,:,:],label,model_name)


    '''
    
        opt = keras.optimizers.Adam(learning_rate=0.05)
    loss = 'mse'
    #loss= keras.losses.MeanSquaredLogarithmicError()
    #loss = keras.losses.CosineSimilarity(axis=1)
    #loss = keras.losses.LogCosh()
    autoencoder.compile(optimizer=opt, loss=loss, metrics=['mse'])'''