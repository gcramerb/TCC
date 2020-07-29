from keras.layers import Input, Dense,LSTM,RepeatVector,TimeDistributed, Bidirectional
#from keras.layers.wrappers import Bidirectional
from keras.models import Model, load_model,Sequential
from keras.utils import plot_model
import numpy as np
#import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from keras import backend as K


# 23/05/2019
# criando um LSTM auto encoder

np.random.seed(12227)

#data_input_file = '/home/cramer/Documentos/LabSense/Datasets/LOSO/USCHAD.npz'
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

#activity1_train = np.expand_dims(np.squeeze(activity1_train), axis=-1)
#activity1_test = np.expand_dims(np.squeeze(activity1_test), axis=-1)

activity1_test = np.squeeze(activity1_test)
activity1_train = np.squeeze(activity1_train)


LATENT_SIZE = 100
shape_ac1 = activity1_train.shape
SEQUENCE_LEN  =shape_ac1[1]
EMBED_SIZE = shape_ac1[2]

inputs = Input(shape=(SEQUENCE_LEN, EMBED_SIZE), name="input")
encoded = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum", name="encoder_lstm")(inputs)
decoded = RepeatVector(SEQUENCE_LEN, name="repeater")(encoded)
decoded = Bidirectional(LSTM(EMBED_SIZE, return_sequences=True), merge_mode="sum", name="decoder_lstm")(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer="sgd", loss='mse')
# autoencoder.compile(optimizer="sgd", loss='mse', sample_weight_mode='temporal')
# autoencoder.compile(optimizer="sgd", loss='mse')
# train the model for 10 epochs and save the best model based on MSE loss
#

EPOCHS  =800
BATCH_SIZE = 256
num_train_steps = len(activity1_train) // BATCH_SIZE
num_test_steps = len(activity1_test) // BATCH_SIZE
autoencoder.fit(activity1_train,activity1_train,epochs= EPOCHS, verbose=0)


# demonstrate prediction

autoencoder.save("senConvAutoE3eixo_v1.h5")


'''

decoded_Acc = autoencoder.predict(activity1_test, verbose=0)
plt.figure(1)
plt.subplot(211)
plt.title('Acc teste')
plt.plot(activity1_test[43,:,2])

plt.subplot(212)
plt.title('Acc reconstruido')
plt.plot(decoded_Acc[43,:,2])
plt.show()


'''
