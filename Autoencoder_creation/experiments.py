import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
import json
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from convAE import MyAutoEncoder
import sys
sys.path.insert(0, "../")
from dataHandler import dataHandler
from utils import saveAll



batch_list = [8,16,32]
lr_list = [0.001,0.0005,0.0001]
missing_list = ['0.7']
# 30 de missing:



# 30,50 de missing:
batch = 16
lr = 0.0005
#m = '0.5'
i = 35
for m in missing_list:
	i = i +1
	hyp = dict()
	hyp['conv_window']  = (5, 3)
	hyp['pooling_window'] = (2, 1)
	hyp['n_filters'] = (32, 16, 8)
	hyp['n_epoches']  = 300
	hyp['batch_size'] = batch
	hyp['lr'] = lr


	result_list = []
	curva_list = []
	config = hyp
	exp = f'experiment_{i}'
	path = '../../resultados/'
	path_file = path + exp + '/'
	dirname = os.path.dirname(__file__)
	filename = os.path.join(dirname, path_file)
	DH = dataHandler()
	DH.load_data(dataset_name='UTD-MHAD1_1s.npz', sensor_factor='1.1')

	DH.apply_missing(missing_factor =m,missing_sensor = '1.0')
	DH.impute('mean')

	DH.splitTrainTest()
	X_trainRec = DH.dataXreconstructedTrain
	X_train = DH.dataXtrain
	X_test = DH.dataXreconstructedTest
	trainRec = []
	train = []
	test = []
	for sTrainRec,sTrain in zip(X_trainRec,X_train):
		s0 = np.expand_dims(sTrainRec,axis = 1)
		s1 = np.expand_dims(sTrain,axis = 1)
		trainRec.append(s0)
		train.append(s1)
	for sTest in X_test:
		s2 = np.expand_dims(sTest,axis = 1)
		test.append(s2)

	dim = train[0].shape
	inputs_keras = []

	AE = MyAutoEncoder(hyp)
	AE.buildModel(n_sensors = len(X_train),dim = dim)
	AE.fit(trainRec,train[0])
	x_hat = AE.autoencoder.predict(test)

	curva = AE.history.history
	result = DH.eval_result(x_hat)
	atvList = pd.unique(DH.dataYtest)
	try:
		os.mkdir(filename,0o755)
	except:
		a = 1

	for atv in atvList:
		sample = DH.dataYtest.index(atv)
		pred = x_hat[sample,0,:,:]
		#name = f'TesteComMissing_{sample}'
		DH.plot_result(pred,filename,sample,m)

	saveAll(filename, result, config, curva,m)
	del DH
	del AE
