import numpy as np
import os
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys
import random
import custom_model as cm
import missing_creator as mc
import imput_creator as ic


import keras
from keras import backend as K
from keras.models import load_model

K.set_image_data_format('channels_first')


def _stream(inp, n_filters, kernel, n_classes):
    hidden = keras.layers.Conv2D(
        filters=n_filters[0], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
        padding='same')(inp)
    hidden = keras.layers.MaxPooling2D(pool_size=(2, 1))(hidden)
    hidden = keras.layers.Conv2D(
        filters=n_filters[1], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
        padding='same')(hidden)
    hidden = keras.layers.MaxPooling2D(pool_size=(2, 1))(hidden)

    hidden = keras.layers.Flatten()(hidden)

    activation_dense = 'selu'
    kernel_init_dense = 'glorot_normal'
    n_neurons = 200
    dropout_rate = 0.1

    # -------------- second hidden FC layer --------------------------------------------
    if kernel_init_dense == "":
        hidden = keras.layers.Dense(n_neurons)(hidden)
    else:
        hidden = keras.layers.Dense(n_neurons, kernel_initializer=kernel_init_dense)(hidden)

    hidden = activation_layer(activation_dense, dropout_rate, hidden)

    # -------------- output layer --------------------------------------------

    hidden = keras.layers.Dense(n_classes)(hidden)
    out = keras.layers.core.Activation('softmax')(hidden)

    return out

    # pylint: disable=R0201


def activation_layer(activation, dropout_rate, tensor):
    """Activation layer"""
    import keras
    if activation == 'selu':
        hidden = keras.layers.core.Activation(activation)(tensor)
        hidden = keras.layers.normalization.BatchNormalization()(hidden)
        hidden = keras.layers.noise.AlphaDropout(dropout_rate)(hidden)
    else:
        hidden = keras.layers.core.Activation(activation)(tensor)
        hidden = keras.layers.normalization.BatchNormalization()(hidden)
        hidden = keras.layers.core.Dropout(dropout_rate)(hidden)
    return hidden


def _kernelmlfusion(n_classes, inputs, kernel_pool):
    width = (16, 32)

    streams_models = []
    for inp in inputs:
        for i in range(len(kernel_pool)):
            streams_models.append(_stream(inp, width, kernel_pool[i], n_classes))

    if len(kernel_pool) > 1:
        concat = keras.layers.concatenate(streams_models, axis=-1)
    else:
        concat = streams_models[0]

    hidden = activation_layer('selu', 0.1, concat)
    # -------------- output layer --------------------------------------------

    hidden = keras.layers.Dense(n_classes)(hidden)
    out = keras.layers.core.Activation('softmax')(hidden)

    # -------------- model buider  --------------------------------------------
    model = keras.models.Model(inputs=inputs, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSProp',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    np.random.seed(12227)

    if len(sys.argv) > 1:
        data_input_file = sys.argv[1]
        batch = 128
        sensor_factor = sys.argv[2]
        missing_type = sys.argv[3]
        impute_type = sys.argv[4]

    else:
        #data_input_file = 'MHEALTH.npz'
        data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\MHEALTH.npz'
        batch = 128
        sensor_factor = '1.0.0' # acc
        missing_type = 'b'
        missing = True
        impute_type = 'knn'
    missing_factor_list = [0.05,0.1,0.2,0.25,0.35]
    avg_acc = []
    avg_recall = []
    avg_f1 = []
    avg_f1_fold = []
    avg_acc_fold= []
    avg_recall_fold= []
    # ----------------------------variables of model -----------------

    pool = [(2, 2), (3, 3), (5, 2), (12, 2), (25, 2)]
    X_2k,y,folds = mc.load_data(data_input_file = data_input_file,
                                sensor_factor=sensor_factor,
                                missing_type = missing_type,
                                missing = False,
                                normalize = False)
    n_class = y.shape[1]
    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]
        X_train = []
        X_test = []
        X_train = X_2k[train_idx]
        X_test = X_2k[test_idx]

        acc_miss = []
        recall_miss = []
        f1_miss = []
        for miss_fac in missing_factor_list:
            X_test,idx_missing_all = mc.apply_missing_data(X_test.copy(),missing_type,miss_fac)
            X_test_2 = ic.impute(X_test,impute_type, idx_missing_all)

            inputs = []

            inputs.append(keras.layers.Input((X_2k.shape[1], X_2k.shape[2], X_2k.shape[3])))

            _model = _kernelmlfusion(n_class, inputs, pool)
            _model.fit(X_train, y[train_idx], batch, cm.n_ep, verbose=0,
                       callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)],
                       validation_data=(X_train, y[train_idx]))

            # Your testing goes here. For instance:
            y_pred = _model.predict(X_test)

            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y[test_idx], axis=1)



            acc_miss_ = accuracy_score(y_true, y_pred)
            acc_miss.append(acc_miss_)
            recall_miss_ = recall_score(y_true, y_pred, average='macro')
            recall_miss.append(recall_miss_)

            f1_miss_ = f1_score(y_true, y_pred, average='macro')
            f1_miss.append(f1_miss_)
        avg_f1_fold.append(f1_miss)
        avg_acc_fold.append(acc_miss)
        avg_recall_fold.append(recall_miss)

    for i in range(len(missing_factor_list)):
        ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc_fold[i]), scale=st.sem(avg_acc_fold[i]))
        ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall_fold[i]), scale=st.sem(avg_recall_fold[i]))
        ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1_fold[i]), scale=st.sem(avg_f1_fold[i]))


        print('Mean Accuracy {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_acc_fold[i]), ic_acc[0], ic_acc[1]))
        print('Mean Recall {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_recall_fold[i]), ic_recall[0], ic_recall[1]))
        print('Mean F1 {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_f1_fold[i]), ic_f1[0], ic_f1[1]))
