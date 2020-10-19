import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from keras import backend as K

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from scipy import fftpack


K.set_image_data_format('channels_first')

class freq_Domain():
    def __init__(self):
        self.data_x = None
        self.data_y = None
        self.folds = None
        self.n_class = None
        self.model = None
        self.label_names = None
        self.x_t = []
        self.x_t_i = []
    def set_data(self,dataset_name='MHEALTH'):
        import sys
        sys.path.insert(0, 'C:\\Users\gcram\Documents\GitHub\TCC\TCC\\')
        from utils import get_data, plot_sensor

        data_x, Y, self.folds, self.label_names = get_data(dataset_name)
        self.data_x = np.array(data_x[:,0,:,0])
        self.data_y = np.array([np.argmax(y) for y in Y])
        self.n_class = Y.shape[1]

    def transform(self):
        #mhealth: f = 50Hz

        f = 50  # Frequency, in cycles per second, or Hertz
        f_s = 50  # Sampling rate, or number of measurements per second


        for i in range(len(self.data_x)):
            self.x_t.append(fftpack.fft(self.data_x[i,:]))



    def inv_transform(self):

        for i in range(len(self.x_t)):
            inv_t = fftpack.ifft(self.x_t[i])
            inv_t = np.abs(inv_t)
            self.x_t_i.append(inv_t)

    def plot_reconstruction(self,index):
        f, axarr = plt.subplots(2, sharex=True, sharey=True)
        label = self.label_names[index]
        axarr[0].plot(self.data_x[index,:], color='green', label='x')
        axarr[0].set_title('ACC Original - {}'.format(label))
        axarr[0].legend()

        axarr[1].plot(self.x_t_i[index], color='red', label='x')
        axarr[1].set_title('ACC reconstructed')
        axarr[1].legend()
        plt.show()


