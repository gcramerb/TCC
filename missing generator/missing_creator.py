import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def apply_missing_data(data,missing_type,missing_factor,normalize = False):
    dim = data.shape[2]
    if missing_type == 'b':
        block_range = round(dim * float(missing_factor))
        idx_range_max = dim - 1 - block_range
        idx_missing_all = []
        for i in range(data.shape[0]):
            idx_missing = random.sample(range(0, idx_range_max), 1)[0]
            data[i, :, idx_missing:idx_missing + block_range, 0:3] = np.nan
            idx_missing_all.append(range(idx_missing,idx_missing + block_range))

    if missing_type == 'nb':
        # usamos valor defaut de 3 partes ausentes
        # a princípo não está sendo tratado se os blocos faltantes forem sobrepostos.
        n = 3
        block_range = round(dim * float(missing_factor))
        idx_range_max = dim - 1 - block_range
        for i in range(n):
            for i in range(data.shape[0]):
                idx_missing = random.sample(range(0, idx_range_max), 1)[0]
                data[i, :, idx_missing:idx_missing + block_range, 0:3] = np.nan


    elif missing_type == 'u':
        idx_missing = random.sample(range(0, dim), round(dim * float(missing_factor)))
        for i in idx_missing:
            data[:, :, i, 0:3] = np.nan
    return data,idx_missing_all

def load_data(data_input_file, missing_factor = '0.2',sensor_factor= '1.0.0',missing_type='b',missing = False,normalize = False):
    tmp = np.load(data_input_file, allow_pickle=True)
    X = tmp['X']
    y = tmp['y']
    folds = tmp['folds']
    X_modas = []
    print('sensor factor: ')
    print(sensor_factor)

    dataset_name = data_input_file.split('\\')[-1]
    print(dataset_name)
    if dataset_name == 'MHEALTH.npz':
        data = []
        temp = []
        data.append(X[:, :, :, 14:17])  # ACC right-lower-arm
        data.append(X[:, :, :, 17:20])  # GYR right-lower-arm
        data.append(X[:, :, :, 20:23])  # MAG right-lower-arm
        s = sensor_factor.split('.')
        for i in range(len(s)):
            if s[i] == '1':
                temp.append(data[i])

    if dataset_name == 'PAMAP2P.npz':
        data = []
        temp = []
        sensor_location = '3'
        if sensor_location == '1':
            data.append(X[:, :, :, 1:4])  # ACC2 right-lower-arm
            data.append(X[:, :, :, 7:10])  # GYR2 right-lower-arm
            data.append(X[:, :, :, 10:13])  # MAG2 right-lower-arm
        if sensor_location == '2':
            data.append(X[:, :, :, 17:20])  # ACC2 right-lower-arm
            data.append(X[:, :, :, 20:23])  # GYR2 right-lower-arm
            data.append(X[:, :, :, 23:26])  # MAG2 right-lower-arm
        if sensor_location == '3':
            data.append(X[:, :, :, 27:30])  # ACC2 right-lower-arm
            data.append(X[:, :, :, 33:36])  # GYR2 right-lower-arm
            data.append(X[:, :, :, 36:39])  # MAG2 right-lower-arm
        s = sensor_factor.split('.')

        for i in range(len(s)):
            if s[i] == '1':
                temp.append(data[i])

    if dataset_name == 'UTD-MHAD1_1s.npz':
        data = []
        temp = []
        data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
        data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
        s = sensor_factor.split('.')
        for i in range(len(s)):
            if s[i] == '1':
                temp.append(data[i])

    if dataset_name == 'UTD-MHAD2_1s.npz':
        data = []
        temp = []
        data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
        data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
        s = sensor_factor.split('.')
        for i in range(len(s)):
            if s[i] == '1':
                temp.append(data[i])

    if dataset_name == 'WHARF.npz':
        data = []
        temp = []
        data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
        s = sensor_factor.split('.')
        for i in range(len(s)):
            if s[i] == '1':
                temp.append(data[i])

    if dataset_name == 'USCHAD.npz':
        data = []
        temp = []
        data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
        data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
        s = sensor_factor.split('.')
        for i in range(len(s)):
            if s[i] == '1':
                temp.append(data[i])

    if dataset_name == 'WISDM.npz':
        data = []
        temp = []
        data.append(X[:, :, :, 3:6])  # ACC right-lower-arm
        s = sensor_factor.split('.')
        for i in range(len(s)):
            if s[i] == '1':
                temp.append(data[i])

    X_2k = np.concatenate(temp, axis=-1)
    if normalize:
        for ii in range(X_2k.shape[0]):
            for jj in range(X_2k.shape[1]):
                scaler = MinMaxScaler()
                scaler.fit(X_2k[ii,jj,:,0:3])
                X_2k[ii,jj,:,0:3] = (scaler.transform(X_2k[ii,jj,:,0:3]))* 256


    if missing:
        dim = X_2k.shape[2]
        if missing_type ==  'b':
            block_range = round(dim*float(missing_factor))
            idx_range_max = dim - 1 - block_range
            idx_missing1 = random.sample(range(0, idx_range_max), 1)[0]
            idx_missing2 = random.sample(range(0, idx_range_max), 1)[0]
            idx_missing3 = random.sample(range(0, idx_range_max), 1)[0]
            X_2k[:,:,idx_missing1:idx_missing1+block_range,0:3] = -1
            X_2k[:, :, idx_missing2:idx_missing2 + block_range, 0:3] = -1
            X_2k[:, :, idx_missing3:idx_missing3 + block_range, 0:3] = -1


        elif missing_type == 'u':
            idx_missing = random.sample(range(0, dim), round(dim * float(missing_factor)))
            for i in idx_missing:
                X_2k[:, :, i, 0:3] = -1

    return X_2k,y,folds
