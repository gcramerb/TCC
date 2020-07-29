from keras.models import load_model
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
#from matplotlib import pyplot



#--------------------------------------------------------------------------------------------------------------------
data_input_file = '/home/cramer/Documentos/LabSense/Datasets/LOSO/USCHAD.npz'
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
if dataset_name == 'UTD-MHAD2_1s.npz' or dataset_name == 'UTD-MHAD1_1s.npz' or dataset_name == 'USCHAD.npz':
	X_One = X[:, :, :, 0:3]
elif dataset_name == 'WHARF.npz' or dataset_name == 'WISDM.npz':
	X_One = X[:, :, :, 0:3]
elif dataset_name == 'MHEALTH.npz':
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

activity1_test = np.squeeze(activity1_test)
activity1_train = np.squeeze(activity1_train)


autoencoder = load_model("lasm_autoV2.h5")


decoded_Acc = autoencoder.predict(activity1_train)
activity1_test =  np.squeeze(activity1_test)
decoded_Acc = np.squeeze(decoded_Acc)

a= 10

#trainX_walk = trainX_walk.reshape((len(trainAccX_walk),128,9 ))

a= 10
plt.figure(1)
plt.subplot(211)
plt.title('Acc teste')
plt.plot(activity1_train[43,:,0])

plt.subplot(212)
plt.title('Acc reconstruido')
plt.plot(decoded_Acc[43,:,0])
plt.show()