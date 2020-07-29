from keras.layers import Input, Dense, UpSampling2D,Conv2D, MaxPooling2D
from keras.models import Model, load_model
import numpy as np
from pandas import read_csv, DataFrame
from keras import backend as K


#if __name__ == '__main__':
np.random.seed(12227)

data_input_file = '/home/guilherme.silva/datasets/LOSO/USCHAD.npz'
tmp = np.load(data_input_file, allow_pickle=True)
X = tmp['X']
# For sklearn methods X = X[:, 0, :, :]
y = tmp['y']
folds = tmp['folds']

X_modas = []

	# ------------------------------------------------------------------------------------
	# split dataset into modalities
	# ------------------------------------------------------------------------------------
dataset_name = data_input_file.split('/')[-1]
# lendo apenas o Acc - os eixos x,y,z :
if dataset_name == 'UTD-MHAD2_1s.npz' or dataset_name == 'UTD-MHAD1_1s.npz' or dataset_name == 'USCHAD.npz':
	X_One = X[:, :, :, 0:3]
elif dataset_name == 'WHARF.npz' or dataset_name == 'WISDM.npz':
	X_One = X[:, :, :, 0:3]
elif dataset_name == 'PAMAP2P.npz':
	X_One = X[:, :, :, 0:3]


n_class = y.shape[1]
	# for i in range(0, len(folds)):
i = 0
train_idx = folds[i][0]
test_idx = folds[i][1]
X_train = []
X_test = []

X_train = X_One[train_idx]
X_test = X_One[test_idx]
Y_train = [np.argmax(y) for y in y[train_idx]]
Y_test = [np.argmax(y) for y in y[test_idx]]

activity1_train = []
activity1_test = []
ativ_id = 2
for i in range(X_train.shape[0]):
	if Y_train[i] == ativ_id:
		activity1_train.append(X_train[i])

for i in range(X_test.shape[0]):
	if Y_test[i] == ativ_id:
		activity1_test.append(X_test[i])

activity1_train = np.expand_dims(np.squeeze(activity1_train), axis=-1)
activity1_test = np.expand_dims(np.squeeze(activity1_test), axis=-1)

	#--------------------------------------------------------------------------------------------------------------------
a = 10

input_img = Input(shape=(activity1_train.shape[1], activity1_train.shape[2],1))  # adapt this if using `channels_first` image data format



x1 = Conv2D(32, (100,3), activation='relu', padding='same',data_format="channels_last")(input_img)
x2 = MaxPooling2D((2,1), padding='same')(x1)
x3 = Conv2D(16, (100,3), activation='relu', padding='same',data_format="channels_last")(x2)
encoded = MaxPooling2D((2,1), padding='same')(x3)


x4 = Conv2D(16, (100,3), activation='relu', padding='same',data_format="channels_last")(encoded)
x5 = UpSampling2D((2,1))(x4)
x6 = Conv2D(32, (100,3), activation='relu', padding='same',data_format="channels_last")(x5)
x7 = UpSampling2D((2,1))(x6)
decoded = Conv2D(1, (2,3), activation='sigmoid', padding='same',data_format="channels_last")(x7)



autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics= ['mse','accuracy'])

autoencoder.fit([activity1_train], [activity1_train], epochs=400, batch_size=512,shuffle=True, validation_data=([activity1_train], [activity1_train]))
autoencoder.save("senConvAutoE3eixov3.h5")