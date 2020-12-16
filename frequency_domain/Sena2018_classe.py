import numpy as np
import pandas as pd

from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys
import custom_model as cm
#import cv2

import random

import keras
from keras import backend as K
from keras.models import load_model

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools

K.set_image_data_format('channels_first')
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)




from sklearn.metrics import confusion_matrix

class modelSena:
	def __init__(self, dim):

		self.model = None
		self.pool = [(2, 2), (3, 3), (5, 2), (12, 2), (25, 2)]
		self.n_class = 12
		inputs = []
		for x in range(dim):
			inputs.append(keras.layers.Input((1, 250, 3)))
		self.inputs = inputs

		np.random.seed(12227)

	def _stream(self,inp, n_filters, kernel, n_classes):
		hidden = keras.layers.Conv2D(
			filters=n_filters[0], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
			padding='same')(inp)
		hidden = keras.layers.MaxPooling2D(pool_size=(2, 1))(hidden)
		hidden = keras.layers.Conv2D(
			filters=n_filters[1], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
			padding='same')(hidden)
		#print('shape - para max pool')
		#print(hidden.shape)
		hidden = keras.layers.MaxPooling2D(pool_size=(2, 1))(hidden)

		hidden = keras.layers.Flatten()(hidden)

		activation_dense = 'selu'
		kernel_init_dense = 'glorot_normal'
		n_neurons = 200
		dropout_rate = 0.1

		# -------------- second hidden FC layer --------------------------------------------
		if kernel_init_dense == "":
			hidden = keras.layers.Dense(n_neurons)(hidden)
		else:
			hidden = keras.layers.Dense(n_neurons, kernel_initializer=kernel_init_dense)(hidden)

		hidden = self.activation_layer(activation_dense, dropout_rate, hidden)

		# -------------- output layer --------------------------------------------

		hidden = keras.layers.Dense(n_classes)(hidden)
		out = keras.layers.core.Activation('softmax')(hidden)
		return out

		# pylint: disable=R0201


	def activation_layer(self,activation, dropout_rate, tensor):
		"""Activation layer"""
		import keras
		if activation == 'selu':
			hidden = keras.layers.core.Activation(activation)(tensor)
			hidden = keras.layers.normalization.BatchNormalization()(hidden)
			hidden = keras.layers.noise.AlphaDropout(dropout_rate)(hidden)
		else:
			hidden = keras.layers.core.Activation(activation)(tensor)
			hidden = keras.layers.normalization.BatchNormalization()(hidden)
			hidden = keras.layers.core.Dropout(dropout_rate)(hidden)
		return hidden


	def _kernelmlfusion(self,n_classes, inputs, kernel_pool):
		width = (16, 32)
		streams_models = []
		for inp in inputs:
			for i in range(len(kernel_pool)):
				streams_models.append(self._stream(inp, width, kernel_pool[i], n_classes))

		# pasa o input -1 pra um mlp ou não
		# passa um softmax
		# streams_models append o softmax

		if len(kernel_pool) > 1:
			concat = keras.layers.concatenate(streams_models, axis=-1)
		else:
			concat = streams_models[0]

		hidden = self.activation_layer('selu', 0.1, concat)
		# -------------- output layer --------------------------------------------

		hidden = keras.layers.Dense(n_classes)(hidden)
		out = keras.layers.core.Activation('softmax')(hidden)

		# -------------- model buider  --------------------------------------------
		model = keras.models.Model(inputs=inputs, outputs=out)
		model.compile(loss='categorical_crossentropy',
					  optimizer='RMSProp',
					  metrics=['accuracy'])

		return model

	def build(self):

		self.model = self._kernelmlfusion(self.n_class, self.inputs, self.pool)

	def fit(self,xTrain,yTrain):
		self.model.fit(xTrain, yTrain, 64,200, verbose=0,
				   callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)],
				   validation_data=(xTrain,yTrain))
	def save(self,filename):
		self.model.save(filename)


	def predict(self,xTest):
		y_pred = self.model.predict(xTest)
		return y_pred




from frequency_domain_reconstruction import freq_Domain
fd = freq_Domain()
fd.set_data(dataset_name='MHEALTH')
X, y = np.expand_dims(fd.data_x, axis=1), fd.data_y
clf = modelSena(len(X))
clf.build()
clf.fit()
clf.save('myModel1.h5')


