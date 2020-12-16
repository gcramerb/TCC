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
import sys

sys.path.insert(0, 'C:\\Users\gcram\Documents\GitHub\TCC\TCC\\')
from utils import dataHandler

K.set_image_data_format('channels_first')

class freq_Domain():
    def __init__(self):
        self.data_x = None
        self.data_x_miss = None
        self.data_y = None
        self.folds = None
        self.n_class = None
        self.model = None
        self.label_names = None
        self.x_t = []
        self.x_t_i = []

    def set_data(self,dataset_name='MHEALTH'):


        data_x, Y, self.folds, self.label_names = get_data(dataset_name,missing = False)
        data_x_missing, _, _, _ = get_data(dataset_name, missing=True)
        self.data_x = np.array(data_x[:,0,:,:])
        self.data_x_missing = np.array(data_x_missing[:, 0, :, 0])
        self.data_y = np.array([np.argmax(y) for y in Y])
        self.n_class = Y.shape[1]

    def impute(self,impute_type = 'default'):
        self.data_x_missing = ic.impute(self.data_x_missing,impute_type)





    def transform(self):
        #mhealth: f = 50Hz

        f = 50  # Frequency, in cycles per second, or Hertz
        f_s = 50  # Sampling rate, or number of measurements per second


        for i in range(len(self.data_x)):
            self.x_t.append(fftpack.rfft(self.data_x[i,:]))



    def inv_transform(self):

        for i in range(len(self.x_t)):
            inv_t = fftpack.irfft(self.x_t[i])
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

        def plot_fft(s, fft):
            T = 0.02
            N = s.size
            # f = np.linspace(0, 1 / T, N)
            # fornece os componentes de frequência correspondentes aos dados
            f = np.fft.fftfreq(len(s), T)
            frequencias = f[:N // 2]
            amplitudes = np.abs(fft)[:N // 2] * 1 / N

            return frequencias, amplitudes

            # Plot the signal
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            ax1.plot(x_true, label='acc. X')
            ax1.title.set_text('norm ortho acc. X ')
            # dct_x = dct(ax, norm='ortho')
            dct_x = x_f_true
            f, a = plot_fft(x_true, dct_x)
            ax2.bar(f, a, width=1.5)
            ax2.title.set_text('dct acc. X')
            ax3.plot(fftpack.irfft(dct_x))
            ax3.title.set_text('inverse dct acc. X')
            plt.subplots_adjust(hspace=0.8)

            plt.show()



