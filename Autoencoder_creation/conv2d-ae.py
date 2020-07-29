import numpy as np
import sys
import keras
from keras.layers import Input, Flatten, Reshape
from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import MaxPooling2D, UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

K.set_image_data_format('channels_first')

def autoencoder_ex(shape):

    # Encoder Layers
    inp = Input(shape=shape)
    x1 = Conv2D(32, (15, 1), activation='relu', padding='same')(inp)
    x2 = MaxPooling2D((2, 1), padding='same')(x1)
    
    x3 = Conv2D(16, (15, 1), activation='relu', padding='same')(x2)
    x4 = MaxPooling2D((2, 1), padding='same')(x3)
    
    x5 = Conv2D(16, (15, 1), strides=(5,1), activation='relu', padding='same')(x4)

    ax5 = Conv2D(16, (15, 1), strides=(5, 1), activation='relu', padding='same')(x5)
    
    # Flatten encoding for visualization
    #x = Flatten()(x)
    #x = Reshape((4, 4, 8))(x)
    ax6 = Conv2D(16, (15, 1), activation='relu', padding='same')(ax5)
    ax7 = UpSampling2D((5, 1))(ax6)

    # Decoder Layers
    x6 = Conv2D(16, (15, 1), activation='relu', padding='same')(ax7)
    x7 = UpSampling2D((5, 1))(x6)
    
    x8 = Conv2D(16, (15, 1), activation='relu', padding='same')(x7)
    x9 = UpSampling2D((2, 1))(x8)
    
    x10 = Conv2D(32, (15, 1), activation='relu', padding='same')(x9)
    x11 = UpSampling2D((2, 1))(x10)
    
    x12 = Conv2D(1, (15, 1), activation='sigmoid', padding='same')(x11)
    
    #autoencoder.summary()

    return Model(inp, x12)



def encoder_conv_block(inp, filters):
    e1 = Conv2D(filters, (15,3), activation='relu', padding='same')(inp)
    e2 = MaxPooling2D((2,1))(e1)
    e3 = BatchNormalization()(e2)
    e4 = Activation('relu')(e3)
    return e4


def decoder_conv_block(inp, filters):
    d1 = Conv2D(filters, (15,3), activation='relu', padding='same')(inp)
    d2 = UpSampling2D((2,1))(d1)
    d3 = BatchNormalization()(d2)
    d4 = Activation('relu')(d3)
    return d4

def cae(shape):
    inp = Input(shape=shape)
    ae1 = encoder_conv_block(inp, 32)
    ae2 = encoder_conv_block(ae1, 16)
    #ae3 = encoder_conv_block(ae2, 8)

    #ae4 = decoder_conv_block(ae3, 8)
    ae5 = decoder_conv_block(ae2, 16)
    ae6 = decoder_conv_block(ae5, 32)

    out = Conv2D(shape[0], (15,3), activation='sigmoid', padding='same')(ae6)


    return Model(inp, out)


if __name__ == '__main__':
    np.random.seed(12227)


    if len(sys.argv) > 1:
        data_input_file = sys.argv[1]
    else:
        #data_input_file = '/mnt/users/jessica/Pesquisa/sensor_translation/USCHAD_coss5fold_LOSO.npz'
        data_input_file = '/home/guilherme.silva/datasets/LOSO/PAMAP2P_AVSS_coss5fold_LOSO.npz'
        #data_input_file = 'Z:/Pesquisa/sensor_translation/PAMAP2P_AVSS_coss5fold_LOSO.npz'

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
    n_ep = 2000
    
    fold_models = []

    for i in range(0, len(folds)):
        print("FOLD {}/{}".format(i+1,len(folds)))
        sys.stdout.flush()
        train_idx = folds[i][0]
        test_idx = folds[i][1]
        X_train = X[train_idx]
        X_test = X[test_idx]

        ACC_train = X_train[:,:, :, 0:3]
        GYR_train = X_train[:, :, :, 3:6]

        ACC_test = X_test[:,:,:,  0:3]
        GYR_test = X_test[:, :, :, 3:6]

        #cae_model= cae((ACC_train.shape[1], ACC_train.shape[2], ACC_train.shape[3]))
        #Adam(0.0002, 0.5)
        cae_model = autoencoder_ex((ACC_train.shape[1], ACC_train.shape[2], ACC_train.shape[3]))
        cae_model.compile(loss=['mse'], optimizer=Adam(0.0001, 0.9, epsilon=1e-08))
        #cae_model.compile(optimizer='adam', loss='binary_crossentropy')
        history = cae_model.fit(ACC_train, GYR_train, batch_size, n_ep, verbose=2, shuffle=True, validation_data=(ACC_test, GYR_test))
        nome_model = "cae_model_" + str(i+1) + ".h5"
        fold_models.save(nome_model)


        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss fold ' + str(i+1))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.legend(['train'], loc='upper left')
        plt.savefig("../results/{}_conv2dAE_loss_fold[{}].png".format(dataset_name,i+1))
        plt.clf()
        #exit()
        
    np.savez_compressed('../{}_Conv2DAEmodels'.format(dataset_name), models=fold_models)
