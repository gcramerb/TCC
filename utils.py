import pandas as pd
import numpy as np
import random
from scipy import fftpack
from sklearn.preprocessing import MinMaxScaler

from numpy.random import seed
from numpy.random import rand
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



class dataHandler():
	def __init__(self):
		self.dataX = None
		self.dataXmissing = None
		self.dataXreconstructed = None
		self.dataY = None
		self.folds = None
		self.labelsNames = None
		self.nClass = None

	def load_data(self,dataset_name, sensor_factor='1.0.0', normalize=False):
		data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\LOSO\\' + dataset_name + '.npz'
		self.target_names(dataset = dataset_name)

		tmp = np.load(data_input_file, allow_pickle=True)
		X = tmp['X']
		y_ = tmp['y']
		self.nClass = y_.shape[1]
		self.dataY = [np.argmax(i) for i in y_]
		self.folds = tmp['folds']

		dataset_name = data_input_file.split('\\')[-1]


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
					temp.append(data[i])

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

		Xsensor = np.concatenate(temp, axis=-1)

		if normalize:
			for ii in range(Xsensor.shape[0]):
				for jj in range(Xsensor.shape[1]):
					scaler = MinMaxScaler()
					scaler.fit(Xsensor[ii, jj, :, 0:3])
					Xsensor[ii, jj, :, 0:3] = (scaler.transform(Xsensor[ii, jj, :, 0:3])) * 256



		self.dataX = Xsensor



	def target_names(self,dataset):
		class_names = ""
		if dataset == 'MHEALTH':
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

		elif dataset == 'PAMAP2P':

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
		elif dataset == 'UTD-MHAD1_1s':

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
		elif dataset == 'UTD-MHAD2_1s':

			actNameUTDMHAD2 = {
				0: 'jogging',
				1: 'walking',
				2: 'sit to stand',
				3: 'stand to sit',
				4: 'forward lunge',
				5: 'squat'}
			class_names = actNameUTDMHAD2

		elif dataset == 'WHARF':

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

		elif dataset == 'USCHAD':

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
		elif dataset == 'WISDM':

			actNameWISDM = {
				0: 'Jogging',
				1: 'Walking',
				2: 'Upstairs',
				3: 'Downstairs',
				4: 'Sitting',
				5: 'Standing'
			}
			class_names = actNameWISDM
		self.labelNames = class_names

	def apply_missing(self,missing_type = 'b', missing_factor):

		if self.dataX is None:
			print('Dados inexistente ')
			return
		self.dataXmissing = self.dataX.copy()
		dim = self.dataXmissing.shape[1]
		nSamples = self.dataXmissing.shape[0]

		if missing_type == 'b':
			block_range = round(dim * float(missing_factor))
			idx_range_max = dim - 1 - block_range
			idx_missing_all = []
			for i in range(nSamples):
				idx_missing = random.sample(range(0, idx_range_max), 1)[0]
				self.dataXmissing[i, idx_missing:idx_missing + block_range, 0:3] = np.nan

		if missing_type == 'nb':
			# usamos valor defaut de 3 partes ausentes
			# a princípo não está sendo tratado se os blocos faltantes forem sobrepostos.
			n = 3
			block_range = round(dim * float(missing_factor))
			idx_range_max = dim - 1 - block_range
			for i in range(n):
				for i in range(nSamples):
					idx_missing = random.sample(range(0, idx_range_max), 1)[0]
					self.dataXmissing[i, idx_missing:idx_missing + block_range, 0:3] = np.nan


		elif missing_type == 'u':
			idx_missing = random.sample(range(0, dim), round(dim * float(missing_factor)))
			for i in idx_missing:
				self.dataXmissing[:, i, 0:3] = np.nan


	def impute(self,impute_type):
		self.dataXreconstructed = self.dataXmissing.copy()
		nSamples = self.dataXmissing.shape[0]

		dim = self.dataXmissing.shape[1]
		if  impute_type == 'mean':
			for i in range(nSamples):
				idx_missing = np.argwhere(np.isnan(self.dataXmissing[i]))
				idx_missing = idx_missing.flatten()
				idx_notM = list(set(range(dim)) - set(idx_missing))
				defautMeanX = np.mean(self.dataXmissing[i, idx_notM, 0])
				defautMeanY = np.mean(self.dataXmissing[i, idx_notM, 1])
				defautMeanZ = np.mean(self.dataXmissing[i, idx_notM, 2])
				self.dataXreconstructed[i,idx_missing,0:3] = [defautMeanX,defautMeanY,defautMeanZ]
				#defautMeanX = np.mean(data_missing[i, idx_notM])
				#data_missing[i, idx_missing] = defautMeanX

		if  impute_type == 'mode':
			for i in range(self.dataXmissing.shape[0]):
				idx_missing = np.argwhere(np.isnan(self.dataXmissing[i]))
				idx_missing = idx_missing.flatten()
				idx_notM = list(set(range(dim))- set(idx_missing))
				defautModeX = statistics.mode(self.dataXmissing[i,idx_notM,0])
				defautModeY = statistics.mode(self.dataXmissing[i,idx_notM,1])
				defautModeZ = statistics.mode(self.dataXmissing[i,idx_notM,2])
				self.dataXreconstructed[i,idx_missing,0:3] = [defautModeX,defautModeY,defautModeZ]


		if impute_type == 'median':
			for i in range(nSamples):
				idx_missing = np.argwhere(np.isnan(self.dataXmissing[i]))
				idx_missing = idx_missing.flatten()
				idx_notM = list(set(range(dim))- set(idx_missing))
				defautMedianX = np.median(data_missing[i,idx_notM,0])
				defautMedianY = np.median(data_missing[i,idx_notM,1])
				defautMedianZ = np.median(data_missing[i,idx_notM,2])
				self.dataXreconstructed[i,idx_missing,0:3] = [defautMedianX,defautMedianY,defautMedianZ]

		if impute_type == 'last_value':
			for i in range(nSamples):
				idx_missing = np.argwhere(np.isnan(self.dataXmissing[i]))
				idx_missing = idx_missing.flatten()

				idx_notM = list(set(range(dim)) - set(idx_missing))
				lastVx = data_missing[i,idx_missing[i][0]-1,0]
				lastVy = data_missing[i,idx_missing[i][0]-1,1]
				lastVz = data_missing[i,idx_missing[i][0]-1,2]
				self.dataXreconstructed[i, idx_missing, 0:3] = [lastVx, lastVy, lastVz]
		if impute_type == 'aleatory':
			seed(22277)
			for i in range(nSamples):
				idx_missing = np.argwhere(np.isnan(self.dataXmissing[i]))
				idx_missing = idx_missing.flatten()
				n = len(idx_missing)
				minX = np.nanmin(self.dataXmissing[i,:,0])
				minY = np.nanmin(self.dataXmissing[i,:,1])
				minZ = np.nanmin(self.dataXmissing[i,:,2])

				maxX = np.nanmax(self.dataXmissing[i,:,0])
				maxY= np.nanmax(self.dataXmissing[i,:,1])
				maxZ =np.nanmax(self.dataXmissing[i,:,2])
				x = minX + (rand(n) * (maxX - minX))
				y = minY + (rand(n) * (maxY - minY))
				z = minZ + (rand(n) * (maxZ - minZ))
				self.dataXreconstructed[i, idx_missing, 0:3] = [x,y,z]



		if impute_type == 'interpolation':
			for i in range(nSamples):
				self.dataXreconstructed[i,:,0 ] = pd.Series(self.dataXmissing[i,:,0 ]).interpolate()
				self.dataXreconstructed[i,:, 1] = pd.Series(self.dataXmissing[i, :, 1]).interpolate()
				self.dataXreconstructed[i,:,2] = pd.Series(self.dataXmissing[i, :, 2]).interpolate()

		if impute_type == 'default':
			for i in range(nSamples):
				idx_missing = np.argwhere(np.isnan(self.dataXmissing[i]))
				idx_missing = idx_missing.flatten()
				self.dataXreconstructed[i, idx_missing, 0:3] = [0, 0, -9.81]
				#self.dataXmissing[i, idx_missing] = 0

		if impute_type == 'frequency':
			for i in range(nSamples):
				idx_missing = np.argwhere(np.isnan(self.dataXmissing[i,:,0]))
				idx_missing = idx_missing.flatten()
				idx_notM = list(set(range(dim)) - set(idx_missing))
				xfreq =  fftpack.rfft(self.dataXmissing[i, idx_notM, 0])
				yfreq =fftpack.rfft(self.dataXmissing[i, idx_notM, 1])
				zfreq =fftpack.rfft(self.dataXmissing[i, idx_notM, 2])

				self.dataXreconstructed[i,idx_missing,0] = fftpack.irfft(xfreq, n=len(idx_missing))
				self.dataXreconstructed[i, idx_missing, 1] = fftpack.irfft(yfreq, n=len(idx_missing))
				self.dataXreconstructed[i, idx_missing, 2] = fftpack.irfft(zfreq, n=len(idx_missing))




class Plots:

	def plot_sensor(true,pred,file_name,label):
		sensor = ['acc','gyr','mag']
		axis = [' x',' y',' z']
		df_true = pd.DataFrame()
		df_pred = pd.DataFrame()
		for i in range(true.shape[-1]):
			col_true = sensor[int(i/3)] + axis[i%3] + ' true'
			pd.concat([df_true,pd.DataFrame(data = true[:,i],columns = [col_true])],axis = 1)
			col_pred = sensor[int(i/3)] + axis[i%3] + ' pred'
			pd.concat([df_pred,pd.DataFrame(data = pred[:,i],columns = [col_pred])],axis = 1)



		#fig = df.plot().get_figure()
		#fig.savefig('mhealth_reconstructed.png')
		#def plot_sensor(acc, acc_rec, label, model_name)
		f, axarr = plt.subplots(2, sharex=True, sharey=True)
		# pyplot.figure()

		# determine the total number of plots
		# n, off = imgs_B.shape[2] + 1, 0
		#sensor = np.squeeze(acc)
		# plot total acc
		axarr[0].plot(true[:, 0], color='green',label = 'x')
		axarr[0].plot(true[:, 1], color='blue',label = 'y')
		axarr[0].plot(true[:, 2], color='red',label = 'z')
		axarr[0].set_title('ACC Original - {}'.format(label))
		axarr[0].legend()
		# plot total acc
		axarr[1].plot(pred[:, 0], color='green',label = 'x')
		axarr[1].plot(pred[:, 1], color='blue',label = 'y')
		axarr[1].plot(pred[:, 2], color='red',label = 'z')
		axarr[1].set_title('ACC reconstructed')
		axarr[1].legend()

		plt.savefig("C:\\Users\gcram\Documents\Github\TCC\images\%s_%s.png" % (label, file_name))
		plt.close()

