from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from keras import backend as K

# funcoes para carregar os dados:
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
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


# carregar os dados por completo:

trainX, trainy = load_dataset('train', 'UCI HAR Dataset/UCI HAR Dataset/')
# load all test
testX, testy = load_dataset('test', 'UCI HAR Dataset/UCI HAR Dataset/')

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
k =0
act_id = activity_ids[k]
act_id_test = activity_ids_test[k]

# acc1 , acc2 , gyr
trainX_walk = (grouped[act_id])
testX_walk = grouped_test[act_id_test]


# vamos pegar apenas o eixo X do Acc1, a atividade em questao:
trainAccX_walk  = trainX_walk[:,:,0]
testAccX_walk = testX_walk[:,:,0]

#trainAccX_walk = trainAccX_walk.reshape((len(trainAccX_walk), np.prod(trainAccX_walk.shape[1:])))
#testAccX_walk = testAccX_walk.reshape((len(testAccX_walk), np.prod(testAccX_walk.shape[1:])))

#trainAccX_walk = np.reshape(trainAccX_walk, (1226,1,128))
trainAccX_walk = np.expand_dims(trainAccX_walk, axis=-1)
#testAccX_walk = np.reshape(testAccX_walk, (496,1,128))
testAccX_walk = np.expand_dims(testAccX_walk, axis=-1)

#--------------------------------------------------------------------------------------------------------------------


input_img = Input(shape=(128,1))  # adapt this if using `channels_first` image data format

x = Conv1D(32, (8), activation='relu', padding='same')(input_img)
x = MaxPooling1D((2), padding='same')(x)
x = Conv1D(16, (8), activation='relu', padding='same')(x)
# x = MaxPooling1D((2), padding='same')(x)
# x = Conv1D(16, (4), activation='relu', padding='same')(x)
encoded = MaxPooling1D((2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

# x = Conv1D(16, (4), activation='relu', padding='same')(encoded)
# x = UpSampling1D((2))(x)
x = Conv1D(16, (8), activation='relu', padding='same')(encoded)
x = UpSampling1D((2))(x)
x = Conv1D(32, (8), activation='relu', padding='same')(x)
x = UpSampling1D((2))(x)
decoded = Conv1D(1, (2), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics= ['mse','accuracy'])

autoencoder.fit(trainAccX_walk, trainAccX_walk, epochs=400, batch_size=512,shuffle=True, validation_data=(testAccX_walk, testAccX_walk))
autoencoder.save("sensorConvAutoencoder.h5")