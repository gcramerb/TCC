from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt






#carregando a base de dados:
(x_train, _), (x_test, _) = mnist.load_data()

# gerando a base de dados com ruido
x_train_d = x_train.astype('float32') / 255.
x_test_d = x_test.astype('float32') / 255.
x_train_d = np.reshape(x_train_d, (len(x_train_d), 28, 28))  # adapt this if using `channels_first` image data format
x_test_d = np.reshape(x_test_d, (len(x_test_d), 28, 28))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train_d + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test_d + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
print(x_train_noisy.shape)
print(x_test_noisy.shape)

x_train_noisy = x_train_noisy.reshape((len(x_train_noisy), np.prod(x_train_noisy.shape[1:])))
x_test_noisy = x_test_noisy.reshape((len(x_test_noisy), np.prod(x_test_noisy.shape[1:])))
print(x_train_noisy.shape)
print(x_test_noisy.shape)

x_train = x_train_d.reshape((len(x_train_d), np.prod(x_train_d.shape[1:])))
x_test = x_test_d.reshape((len(x_test_d), np.prod(x_test_d.shape[1:])))
print(x_train.shape)
print(x_test.shape)




#---------------------------------------------------------------------------------------------------------------------
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))

#parte1:
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)
# Esse eh o modelo de rede neural para a reconstrucao
autoencoder = Model(input_img, decoded)

#---------------------------------------------------------------------------------------------------------------------
# parte 2 : ( criar de outro jeito) ->  criando o encoder e decoder como modelos separados mas estao juntos na variavel autoencoder
# Encoder Model:
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

#Decoder model
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

#---------------------------------------------------------------------------------------------------------------------
# per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#---------------------------------------------------------------------------------------------------------------------

autoencoder.fit(x_train_noisy, x_train,epochs=35,batch_size=256,shuffle=True,validation_data=(x_test, x_test))
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test_noisy)
decoded_imgs = decoder.predict(encoded_imgs)



n =1
input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
plt.figure(figsize=(20, 4))



# display original
ax = plt.subplot(3, n, 1)
plt.imshow(x_test_d[0])
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# mostrar a com ruido :
ax = plt.subplot(3, n, 2)
plt.imshow(x_test_noisy[0].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# display reconstruction
ax = plt.subplot(3, n, 3)
plt.imshow(decoded_imgs[0].reshape(28, 28))
plt.gray()

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
