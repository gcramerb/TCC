import numpy as np
import sys
import keras
from keras.layers import Input
from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import MaxPooling1D, UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU

def encoder_conv_block(inp, filters):
    e1 = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform')(inp)
    e2 = MaxPooling1D(2)(e1)
    e3 = BatchNormalization()(e2)
    e4 = LeakyReLU(alpha=0.2)(e3)
    #e4 = Activation('relu')(e3)
    return e4


def decoder_conv_block(inp, filters):

    
    d1 = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same')(inp)
    d2 = UpSampling1D(2)(d1)
    d3 = BatchNormalization()(d2)
    d4 = LeakyReLU(alpha=0.2)(d3)
    #d4 = Activation('relu')(d3)
    return d4

def cae(shape):
    inp = Input(shape=shape)
    ae1 = encoder_conv_block(inp, 64)
    ae2 = encoder_conv_block(ae1, 32)
    #ae3 = encoder_conv_block(ae2, 8)

    #ae4 = decoder_conv_block(ae3, 8)
    ae5 = decoder_conv_block(ae2, 32)
    ae6 = decoder_conv_block(ae5, 64)

    out = Conv1D(filters=shape[1], kernel_size=3, strides=1, activation= "sigmoid", padding='same')(ae6)


    return Model(inp, out)


if __name__ == '__main__':
    np.random.seed(12227)


    if len(sys.argv) > 1:
        data_input_file = sys.argv[1]
    else:
        data_input_file = '/mnt/users/jessica/Pesquisa/sensor_translation/USCHAD_coss5fold_LOSO.npz'
        #data_input_file = 'Z:/Pesquisa/sensor_translation/USCHAD_coss5fold_LOSO.npz'

    tmp = np.load(data_input_file, allow_pickle=True)
    X = tmp['X']
    y = tmp['y']
    folds = tmp['folds']
    
    dataset_name = data_input_file.split('/')[-1].split('.')[0]

    n_class = y.shape[1]

    avg_acc = []
    avg_recall = []
    avg_f1 = []

    batch_size = 64
    n_ep = 200
    
    fold_models = []

    for i in range(0, len(folds)):
        print("FOLD {}/{}".format(i+1,len(folds)))
        sys.stdout.flush()
        train_idx = folds[i][0]
        test_idx = folds[i][1]
        X_train = X[train_idx]
        X_test = X[test_idx]

        X_train = np.squeeze(X_train)
        ACC_train = X_train[:,:, 0:3]
        GYR_train = X_train[:, :, 3:6]
        
        X_test = np.squeeze(X_test)
        ACC_test = X_test[:,:, 0:3]
        GYR_test = X_test[:, :, 3:6]

        cae_model= cae((ACC_train.shape[1], ACC_train.shape[2]))
        #Adam(0.0002, 0.5)
        #opt = keras.optimizers.Adagrad(lr=0.0001, epsilon=None, decay=0.0)
        #cae_model.compile(loss=['binary_crossentropy'], optimizer=opt)
        cae_model.compile(loss=['mse'], optimizer=Adam(0.0001, 0.9, epsilon=1e-08))
        history = cae_model.fit(ACC_train, GYR_train, batch_size, n_ep, verbose=2, shuffle=True, validation_data=(ACC_test, GYR_test))
        fold_models.append(cae_model)
        
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.legend(['train'], loc='upper left')
        plt.savefig("../results/{}_CAE_loss_fold[{}].png".format(dataset_name,i+1))
        plt.clf()
        #exit()
        
    np.savez_compressed('../{}_CAEmodels'.format(dataset_name), models=fold_models)
