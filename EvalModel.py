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





if __name__ == '__main__':
    np.random.seed(12227)

    data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\MHEALTH.npz'
    batch = 128
    sensor_factor = '1.0.0' # acc
    missing_type = 'b'
    missing = False
    missing_factor_list = [0.05,0.1,0.2,0.25,0.35]

    impute_type = 'median'

    pool = [(2, 2), (3, 3), (5, 2), (12, 2), (25, 2)]


    X,y,folds = mc.load_data(data_input_file = data_input_file,
                                sensor_factor=sensor_factor,
                                missing_type = missing_type,
                                missing = False,
                                normalize = False)
    n_class = y.shape[1]
    for miss_fac in missing_factor_list:
        acc_miss = []
        recall_miss = []
        f1_miss = []

        for i in range(0, len(folds)):
            train_idx = folds[i][0]
            test_idx = folds[i][1]
            X_train = []
            X_test = []
            X_train = X[train_idx]
            X_test = X[test_idx]

            X_test,idx_missing_all = mc.apply_missing_data(X_test.copy(),missing_type,miss_fac)
            X_test = ic.impute(X_test,impute_type, idx_missing_all)

            inputs = []

            # for x in X_modas:
            inputs.append(keras.layers.Input((X.shape[1], X.shape[2], X.shape[3])))
            model_name = 'ModelMHEALTH_fold' + str(i) + '.h5'
            _model = load_model(model_name)


            # Your testing goes here. For instance:
            y_pred = _model.predict(X_test)

            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y[test_idx], axis=1)

            acc = accuracy_score(y_true, y_pred)
            acc_miss.append(acc)

            recall = recall_score(y_true, y_pred, average='macro')
            recall_miss.append(recall)

            f1 = f1_score(y_true, y_pred, average='macro')
            f1_miss.append(f1)

        ic_acc = st.t.interval(0.9, len(acc_miss) - 1, loc=np.mean(acc_miss), scale=st.sem(acc_miss))
        ic_recall = st.t.interval(0.9, len(recall_miss) - 1, loc=np.mean(recall_miss), scale=st.sem(recall_miss))
        ic_f1 = st.t.interval(0.9, len(f1_miss) - 1, loc=np.mean(f1_miss), scale=st.sem(f1_miss))

        arq_name = model_name +impute_type +'_Result.txt'
        arquivo = open(arq_name, 'w')
        arquivo.write('Missing Factor: ', str(miss_fac) )
        arquivo.write('Mean Accuracy {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(acc_miss), ic_acc[0], ic_acc[1]))
        arquivo.write('Mean Recall {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(recall_miss), ic_recall[0], ic_recall[1]))
        arquivo.write('Mean F1 {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(f1_miss), ic_f1[0], ic_f1[1]))
        arquivo.close()



