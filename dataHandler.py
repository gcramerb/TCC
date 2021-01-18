import pandas as pd
import numpy as np
import random
from scipy import fftpack
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_percentage_error

from numpy.random import seed
from numpy.random import rand
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils import classesNames



class dataHandler():
	def __init__(self):
		self.dataX = None
		self.dataXtest =[]
		self.dataXmissing = None
		self.dataXmissingTest = []
		self.dataXreconstructed = None
		self.dataXreconstructedTest = []

		self.dataY = None
		self.dataYtest = None

		self.folds = None
		self.labelsNames = None
		self.nClass = None

		self.imputeType = None
		self.evalResult = None

	def load_data(self,dataset_name, sensor_factor='1.0.0'):
		data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\LOSO\\' + dataset_name
		#data_input_file = '/home/guilherme.silva/datasets/LOSO/' + dataset_name
		tmp = np.load(data_input_file, allow_pickle=True)
		X = tmp['X']
		y_ = tmp['y']
		self.nClass = y_.shape[1]
		self.dataY = [np.argmax(i) for i in y_]
		self.folds = tmp['folds']

		dataset_name = data_input_file.split('\\')[-1]
		self.labelsNames = classesNames(dataset_name)




		if dataset_name == 'MHEALTH.npz':
			data = []
			temp = []
			data.append(X[:, 0, :, 14:17])  # ACC right-lower-arm
			# data.append(X[:, :, :, 5:8])  # ACC left-ankle sensor
			data.append(X[:, 0, :, 17:20])  # GYR right-lower-arm
			data.append(X[:, 0, :, 20:23])  # MAG right-lower-arm

			# data.append(X[:, :, :, 0:3])  # ACC chest-sensor
			# data.append(X[:, :, :, 5:8])  # ACC left-ankle sensor
			# data.append(X[:, :, :, 8:11])   # GYR left-ankle sensor
			# data.append(X[:, :, :, 11:14]) # MAG left-ankle sensor


			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'PAMAP2P.npz':
			data = []
			temp = []
			sensor_location = '3'
			if sensor_location == '1':
				data.append(X[:, 0, :, 1:4])  # ACC2 right-lower-arm
				data.append(X[:, 0, :, 7:10])  # GYR2 right-lower-arm
				data.append(X[:, 0, :, 10:13])  # MAG2 right-lower-arm
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
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'UTD-MHAD1_1s.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'UTD-MHAD2_1s.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'WHARF.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'USCHAD.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'WISDM.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 3:6])  # ACC right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		Xsensor = np.concatenate(temp, axis=-1)
		self.dataX = temp
	def splitTrainTest(self,ratio=0.7):

		samples = len(self.dataX[0])
		np.random.seed(0)
		max_ = int(samples*ratio)
		idx = np.random.permutation(samples)
		idx_train = idx[:max_]
		idx_test =  idx[max_:]
		if self.dataX:
			dataX = deepcopy(self.dataX)
			self.dataXtrain = []
			for sensor in dataX:
				self.dataXtest.append(sensor[idx_test])
				self.dataXtrain.append(sensor[idx_train])
		if self.dataY:
			dataY = deepcopy(self.dataY)
			self.dataYtrain = []
			self.dataYtrain = [dataY[i] for i in idx_train]
			self.dataYtest = [dataY[i] for i in idx_test]
		if self.dataXmissing:
			dataXmissing = deepcopy(self.dataXmissing)
			self.dataXmissingTrain = []
			for sensor in dataXmissing:
				self.dataXmissingTest.append(sensor[idx_test])
				self.dataXmissingTrain.append(sensor[idx_train])
		if self.dataXreconstructed:
			dataXreconstructed = deepcopy(self.dataXreconstructed)
			self.dataXreconstructedTrain = []
			for sensor in dataXreconstructed:
				self.dataXreconstructedTest.append(sensor[idx_test])
				self.dataXreconstructedTrain.append(sensor[idx_train])



	def apply_missing(self,missing_factor,missing_type = 'b',missing_sensor = '1.0.0'):

		if self.dataX is None:
			print('Dados inexistente ')
			return
		self.dataXmissing = deepcopy(self.dataX)
		nSamples = self.dataX[0].shape[0]
		dim = self.dataX[0].shape[1]

		s = missing_sensor.split('.')
		for i in range(len(s)):
			if s[i] == '1':
				if missing_type == 'b':
					block_range = round(dim * float(missing_factor))
					idx_range_max = dim - 1 - block_range
					idx_missing_all = []
					for j in range(nSamples):
						idx_missing = random.sample(range(0, idx_range_max), 1)[0]
						self.dataXmissing[i][j, idx_missing:idx_missing + block_range, 0:3] = np.nan

				if missing_type == 'nb':
					# usamos valor defaut de 3 partes ausentes
					# a princípo não está sendo tratado se os blocos faltantes forem sobrepostos.
					n = 3
					block_range = round(dim * float(missing_factor))
					idx_range_max = dim - 1 - block_range
					for k in range(n):
						for j in range(nSamples):
							idx_missing = random.sample(range(0, idx_range_max), 1)[0]
							self.dataXmissing[i][j, idx_missing:idx_missing + block_range, 0:3] = np.nan


				elif missing_type == 'u':
					idx_missing = random.sample(range(0, dim), round(dim * float(missing_factor)))
					for j in idx_missing:
						sensor[:, j, 0:3] = np.nan


	def impute(self,impute_type):
		self.dataXreconstructed = deepcopy(self.dataXmissing)
		nSamples = self.dataX[0].shape[0]
		dim = self.dataX[0].shape[1]
		self.imputeType = impute_type
		if  impute_type == 'mean':
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0])) #All axis has the same missing points
					idx_missing = idx_missing.flatten()
					idx_notM = list(set(range(dim)) - set(idx_missing))
					defautMeanX = np.mean(sensor[i, idx_notM, 0])
					defautMeanY = np.mean(sensor[i, idx_notM, 1])
					defautMeanZ = np.mean(sensor[i, idx_notM, 2])
					sensor[i,idx_missing,0:3] = [defautMeanX,defautMeanY,defautMeanZ]
					#defautMeanX = np.mean(data_missing[i, idx_notM])
					#data_missing[i, idx_missing] = defautMeanX

		if impute_type == 'median':
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0]))
					idx_missing = idx_missing.flatten()
					idx_notM = list(set(range(dim))- set(idx_missing))
					defautMedianX = np.median(sensor[i,idx_notM,0])
					defautMedianY = np.median(sensor[i,idx_notM,1])
					defautMedianZ = np.median(sensor[i,idx_notM,2])
					sensor[i,idx_missing,0:3] = [defautMedianX,defautMedianY,defautMedianZ]

		if impute_type == 'last_value':
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0]))
					idx_missing = idx_missing.flatten()

					idx_notM = list(set(range(dim)) - set(idx_missing))
					lastVx = sensor[i,idx_missing[i][0]-1,0]
					lastVy = sensor[i,idx_missing[i][0]-1,1]
					lastVz = sensor[i,idx_missing[i][0]-1,2]
					sensor[i, idx_missing, 0:3] = [lastVx, lastVy, lastVz]
		if impute_type == 'aleatory':
			seed(22277)
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0]))
					idx_missing = idx_missing.flatten()
					n = len(idx_missing)
					minX = np.nanmin(sensor[i,:,0])
					minY = np.nanmin(sensor[i,:,1])
					minZ = np.nanmin(sensor[i,:,2])

					maxX = np.nanmax(sensor[i,:,0])
					maxY= np.nanmax(sensor[i,:,1])
					maxZ =np.nanmax(sensor[i,:,2])
					x = minX + (rand(n) * (maxX - minX))
					y = minY + (rand(n) * (maxY - minY))
					z = minZ + (rand(n) * (maxZ - minZ))
					sensor[i, idx_missing, 0:3] = [x,y,z]



		if impute_type == 'interpolation':
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					sensor[i,:,0 ] = pd.Series(sensor[i,:,0 ]).interpolate()
					sensor[i,:, 1] = pd.Series(sensor[i, :, 1]).interpolate()
					senso[i,:,2] = pd.Series(sensor[i, :, 2]).interpolate()

		if impute_type == 'default':
			defalt_values = [[0, 0, -9.81],[0, 0,0],[0, 0, 0]]
			for i in range(nSamples):
				j = 0
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0]))
					idx_missing = idx_missing.flatten()
					sensor[i, idx_missing, 0:3] = defalt_vales[j]
					j = j+1
					#self.dataXmissing[i, idx_missing] = 0

		if impute_type == 'frequency':
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0]))
					idx_missing = idx_missing.flatten()
					idx_notM = list(set(range(dim)) - set(idx_missing))
					xfreq =  fftpack.rfft(sensor[i, idx_notM, 0])
					yfreq =fftpack.rfft(sensor[i, idx_notM, 1])
					zfreq =fftpack.rfft(sensor[i, idx_notM, 2])

					sensor[i,idx_missing,0] = fftpack.irfft(xfreq, n=len(idx_missing))
					sensor[i, idx_missing, 1] = fftpack.irfft(yfreq, n=len(idx_missing))
					sensor[i, idx_missing, 2] = fftpack.irfft(zfreq, n=len(idx_missing))


	def eval_result(self,xpred):

		result = dict()
		result['RMSE'] = dict()
		j = 0
		rmseRec = []
		rmsePred = []
		for axis in ['x','y','z']:
			for i in range(len(self.dataXtest[0])):
				rmsePred.append(mean_squared_error(self.dataXtest[0][i, :, j], xpred[i, 0, :, j], squared=False))
				rmseRec.append(mean_squared_error(self.dataXtest[0][i, :, j], self.dataXreconstructedTest[0][i, :, j], squared=False))
			j = j + 1
			result['RMSE']['autoEncoder  ' + axis] = np.mean(rmsePred)
			result['RMSE']['reconstructed  '+ axis] = np.mean(rmseRec)

		#result['MAPE'] = dict()
		#result['MAPE']['autoEncoder'] = mean_absolute_percentage_error(self.dataXtest[0],xpred)
		#result['MAPE']['reconstructed'] = mean_absolute_percentage_error(self.dataXtest[0],self.dataXreconstructedTest)
		self.evalResult = result
		return result


	def plot_result(self,pred,path,sample=0,tag='teste'):
		sensors = ['acc','gyr','mag']
		axis = [' x',' y',' z']
		true = self.dataXtest[0][sample]
		rec = self.dataXreconstructedTest[0][sample]

		s = self.dataYtest[sample]
		label = self.labelsNames[s]

		f, axarr = plt.subplots(3, sharex=True, sharey=True)
		# pyplot.figure()

		# determine the total number of plots
		# n, off = imgs_B.shape[2] + 1, 0
		#sensor = np.squeeze(acc)
		# plot total TRUE acc
		axarr[0].plot(true[:, 0], color='green',label = 'x')
		axarr[0].plot(true[:, 1], color='blue',label = 'y')
		axarr[0].plot(true[:, 2], color='red',label = 'z')
		axarr[0].set_title('ACC Original - {}'.format(label))
		axarr[0].legend()
		# plot total REconstructed acc
		axarr[1].plot(rec[:, 0], color='green',label = 'x')
		axarr[1].plot(rec[:, 1], color='blue',label = 'y')
		axarr[1].plot(rec[:, 2], color='red',label = 'z')
		axarr[1].set_title('ACC '+ self.imputeType)
		axarr[1].legend()

		# plot total predction acc
		axarr[2].plot(pred[:, 0], color='green',label = 'x')
		axarr[2].plot(pred[:, 1], color='blue',label = 'y')
		axarr[2].plot(pred[:, 2], color='red',label = 'z')
		axarr[2].set_title('ACC Autoencoder ')
		axarr[2].legend()
		#plt.show()

		#plt.savefig(f"C:\\Users\gcram\Documents\Github\TCC\ + folder + '\' {label_file_name}.png")
		file_name = path + f'/{label}_{tag}.png'
		plt.savefig(file_name)
		#plt.savefig("../folder/%s_%s.png" % (label, file_name))

		plt.close()


