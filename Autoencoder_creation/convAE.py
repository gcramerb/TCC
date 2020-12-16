import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout, Conv2D
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
            X_2k[:,:,idx_missing1:idx_missing1+block_range,0:3] = np.nan
            X_2k[:, :, idx_missing2:idx_missing2 + block_range, 0:3] = np.nan
            X_2k[:, :, idx_missing3:idx_missing3 + block_range, 0:3] = np.nan


        elif missing_type == 'u':
            idx_missing = random.sample(range(0, dim), round(dim * float(missing_factor)))
            for i in idx_missing:
                X_2k[:, :, i, 0:3] = np.nan

    return X_2k,y,folds
def target_names(dataset_name):
	class_names = ""
	if dataset_name == 'MHEALTH':
		actNameMHEALTH = {
            0: 'Standing still',
            1: 'Sitting and relaxing',
            2: 'Lying down',
            3: 'Walking',
            4: 'Climbing stairs',
            5: 'Waist bends forward',
            6: 'Frontal elevation\nof arms',
            7: 'Knees bending\n(crouching)',
            8: 'Cycling',
            9: 'Jogging',
            10: 'Running',
            11: 'Jump front and back'
        }
		class_names = actNameMHEALTH

	elif dataset_name == 'PAMAP2P':

		actNamePAMAP2P = {
			0: 'lying',
			1: 'sitting',
			2: 'standing',
			3: 'ironing',
			4: 'vacuum cleaning',
			5: 'ascending stairs',
			6: 'descending stairs',
			7: 'walking',
			8: 'Nordic walking',
			9: 'cycling',
			10: 'running',
			11: 'rope jumping', }
		actNamePAMAP2P_v2 = {
			0: 'Lie',
			1: 'Sit',
			2: 'Stand',
			3: 'Iron',
			4: 'Break',
			5: 'Ascend stairs',
			6: 'Nordic walking',
			7: 'watching TV',
			8: 'computer work',
			9: 'car driving',
			10: 'ascending stairs',
			11: 'descending stairs',
			12: 'vacuum cleaning',
			13: 'ironing',
			14: 'folding laundry',
			15: 'house cleaning',
			16: 'playing soccer',
			17: 'rope jumping',
			18: 'other'}
		class_names = actNamePAMAP2P
	elif dataset_name == 'UTD-MHAD1_1s':

		actNameUTDMHAD = {
			0: 'right arm swipe\nto the left',
			1: 'right arm swipe\nto the right',
			2: 'right hand\nwave',
			3: 'two hand\nfront clap',
			4: 'right arm throw',
			5: 'cross arms\nin the chest',
			6: 'basketball shooting',
			7: 'draw x',
			8: 'draw circle\nclockwise',
			9: 'draw circle\ncounter clockwise',
			10: 'draw triangle',
			11: 'bowling',
			12: 'front boxing',
			13: 'baseball swing\nfrom right',
			14: 'tennis forehand\nswing',
			15: 'arm curl',
			16: 'tennis serve',
			17: 'two hand push',
			18: 'knock on door',
			19: 'hand catch',
			20: 'pick up\nand throw'
		}
		class_names = actNameUTDMHAD
	elif dataset_name == 'UTD-MHAD2_1s':

		actNameUTDMHAD2 = {
			0: 'jogging',
			1: 'walking',
			2: 'sit to stand',
			3: 'stand to sit',
			4: 'forward lunge',
			5: 'squat'}
		class_names = actNameUTDMHAD2

	elif dataset_name == 'WHARF':

		actNameWHARF = {

			0: 'Standup chair',
			1: 'Comb hair',
			2: 'Sitdown chair',
			3: 'Walk',
			4: 'Pour water',
			5: 'Drink glass',
			6: 'Descend stairs',
			7: 'Climb stairs',
			8: 'Liedown bed',
			9: 'Getup bed',
			10: 'Use telephone',
			11: 'Brush teeth'}
		class_names = actNameWHARF

	elif dataset_name == 'USCHAD':

		actNameUSCHAD = {
			0: 'Walking Forward',
			1: 'Walking Left',
			2: 'Walking Right',
			3: 'Walking Upstairs',
			4: 'Walking Downstairs',
			5: 'Running Forward',
			6: 'Jumping Up',
			7: 'Sitting',
			8: 'Standing',
			9: 'Sleeping',
			10: 'Elevator Up',
			11: 'Elevator Down'}
		class_names = actNameUSCHAD
	elif dataset_name == 'WISDM':

		actNameWISDM = {
			0: 'Jogging',
			1: 'Walking',
			2: 'Upstairs',
			3: 'Downstairs',
			4: 'Sitting',
			5: 'Standing'
		}
		class_names = actNameWISDM
	return class_names

def get_data(dataset_name,missing):

    data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\\' + dataset_name + '.npz'
    label_names = target_names(dataset_name)
    batch = 128
    sensor_factor = '1.1.0'
    X, y, folds = load_data(data_input_file=data_input_file,sensor_factor = sensor_factor,missing = missing)
    return X,y,folds,label_names


def myModel(conv_window=(6, 3), pooling_window=(10, 1), n_filters=(64, 32, 16)):
    dims = X_train.shape

    input_img = Input(shape=dims[1:])  # adapt this if using `channels_first` image data format
    print("shape of input", K.int_shape(input_img))
    conv_1 = Conv2D(n_filters[0], conv_window, activation='relu', padding='same')(input_img)
    print("shape after first conv", K.int_shape(conv_1))
    pool_1 = MaxPooling2D(pooling_window, padding='same')(conv_1)
    print("shape after first pooling", K.int_shape(pool_1))
    conv_2 = Conv2D(n_filters[1], conv_window, activation='relu', padding='same')(pool_1)
    print("shape after second conv", K.int_shape(conv_2))

    pool_2 = MaxPooling2D(pooling_window, padding='same')(conv_2)
    print("shape after second pooling", K.int_shape(pool_2))

    conv_3 = Conv2D(n_filters[2], conv_window, activation='relu', padding='same')(pool_2)
    print("shape after third conv", K.int_shape(conv_3))

    encoded = Conv2D(n_filters[2], (9,4), activation='relu', padding='valid')(conv_3)
    #print("shape of conv 4", K.int_shape(conv_4))


    #encoded = MaxPooling2D(pooling_window, padding='same')(conv_4)
    print("shape of encoded", K.int_shape(encoded))

    #encoded_ = MaxPooling2D(pooling_window, padding='valid')(encoded)

    up_3 = UpSampling2D(pooling_window)(encoded)
    print("shape after upsample third pooling", K.int_shape(up_3))

    conv_neg_3 = Conv2D(n_filters[2], conv_window, activation='relu', padding='same')(up_3)
    print("shape after decode third conv", K.int_shape(conv_neg_3))

    up_2 = UpSampling2D(pooling_window)(conv_neg_3)
    print("shape after upsample second pooling", K.int_shape(up_2))

    conv_neg_2 = Conv2D(n_filters[1], conv_window, activation='relu', padding='same')(up_2)
    print("shape after decode second conv", K.int_shape(conv_neg_2))
    up_1 = UpSampling2D(pooling_window)(conv_neg_2)
    print("shape after upsample first pooling", K.int_shape(up_1))
    conv_neg_3 = Conv2D(n_filters[0], conv_window, activation='relu', padding='same')(up_1)
    print("shape after decode first conv", K.int_shape(conv_neg_3))
    decoded = Conv2D(1, conv_window, activation='linear', padding='same')(conv_neg_3)
    print("shape after decode to input", K.int_shape(decoded))

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    #encoder_model = Model(self.autoencoder.input, self.autoencoder.layers[6].output)
    return autoencoder




X,y,folds,label_names = get_data('MHEALTH',missing = False)
X = np.squeeze(X)
i =0
n_class = y.shape[1]
train_idx = folds[i][0]
test_idx = folds[i][1]

X_train = []
X_test = []


#choode only x axis
X_train = np.expand_dims(X[train_idx],axis = 3)
X_test = np.expand_dims(X[test_idx],axis = 3)

y_train = [np.argmax(y) for y in y[train_idx]]
y_test = [np.argmax(y) for y in y[test_idx]]


conv_window=(3, 3)
pooling_window=(5, 1)
n_filters=(32, 16, 8)

autoencoder = myModel(conv_window, pooling_window, n_filters)

n_epoches = 140
batch_size = 8

history = autoencoder.fit(X_train,X_train[:,:,0:3,:], epochs=n_epoches, batch_size=batch_size, shuffle=True)
x_hat = autoencoder.predict(X_test)
print(h_hat)

