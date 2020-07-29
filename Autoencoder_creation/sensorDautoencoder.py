from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
#from matplotlib import pyplot

# funcoes do numpy: dstack array vstack unique

# funcoes para carregar os dados:
# load a list of files, such as x, y, z data for a given variable
# faz a leitura de varios arquivos em uma pasta:
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = np.dstack(loaded)
	return loaded

# faz a leitura de um arquivo:
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values


# load a dataset group, such as train or test
def load_dataset(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# ----------------------------------------------------------------------------------------------------
# group data by activity
def data_by_activity(X, y, activities):
	# group windows by activity
	return {a:X[y[:,0]==a, :, :] for a in activities}

# convert a series of windows to a 1D list
def to_series(windows):
	series = list()
	for window in windows:
		# remove the overlap from the window
		half = int(len(window) / 2) - 1
		for value in window[-half:]:
			series.append(value)
	return series

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# carregar os dados por completo:
# load all train
trainX, trainy = load_dataset('train', 'UCI HAR Dataset/UCI HAR Dataset/')
#print(trainX.shape, trainy.shape)
# load all test
testX, testy = load_dataset('test', 'UCI HAR Dataset/UCI HAR Dataset/')
#print(testX.shape, testy.shape)

# para que fazer isso ?
trainX = trainX.astype('float32') / 255.
testX = testX.astype('float32') / 255.
trainX = np.reshape(trainX, (len(trainX), 128, 9))  # adapt this if using `channels_first` image data format
testX = np.reshape(testX, (len(testX), 128, 9))


# get a list of unique activities for the subject
activity_ids = np.unique(trainy[:,0])
activity_ids_test = np.unique(testy[:,0])
# group windows by activity
grouped = data_by_activity(trainX, trainy, activity_ids)
grouped_test = data_by_activity(testX, testy, activity_ids_test)

# escolhendo apenas uma atividade desse sujeito
k = 0
act_id = activity_ids[k]
act_id_test = activity_ids_test[k]





# acceleracao eixo X Y Z separados :
print((grouped[act_id][:, :,0]).shape)

trainX_walk = (grouped[act_id])
testX_walk = grouped_test[act_id_test]
# vamos pegar apenas o eixo X do Acc, a atividade em questao:
trainAccX_walk  = trainX_walk[:,:,0]
testAccX_walk = testX_walk[:,:,0]

trainAccX_walk = trainAccX_walk.reshape((len(trainAccX_walk), np.prod(trainAccX_walk.shape[1:])))
testAccX_walk = testAccX_walk.reshape((len(testAccX_walk), np.prod(testAccX_walk.shape[1:])))
a = 10
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


# this is the size of our encoded representations
encoding_dim = 64  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(128,))


# Agora, vamos adicionar ruidos aos dados:
#noise_factor = 0.5
#trainAccX_noisy = trainAccX + noise_factor * np.random.normal(loc=0.0, scale=1.0, size= (1226,128))
#testAccX_noisy = testAccX + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=(496,128))


#parte1:
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(15, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(encoding_dim, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(128, activation='sigmoid')(encoded)

# Esse eh o modelo de rede neural para a reconstrucao
autoencoder = Model(input_img, decoded)

#---------------------------------------------------------------------------------------------------------------------
# per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#-----------------------------------------------------------------------------------------------------------------


#We will normalize all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784.


autoencoder.fit(trainAccX_walk,trainAccX_walk,epochs=700,batch_size=256,shuffle=True,validation_data=(testAccX_walk, testAccX_walk),verbose=2)
autoencoder.save("sensorAutoencoder.h5")