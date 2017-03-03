
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.activations import softmax
from keras.objectives import binary_crossentropy as bce

batch_size = 100
data_dim = 784
M = 10
N = 30
nb_epoch = 100
epsilon_std = 0.01

anneal_rate = 0.0003
min_temperature = 0.5

tau = K.variable(5.0, name="temperature")
x = Input(batch_shape=(batch_size, data_dim))
h = Dense(256, activation='relu')(Dense(512, activation='relu')(x))
logits_y = Dense(M*N)(h)

def sampling(logits_y):
    U = K.random_uniform(K.shape(logits_y), 0, 1)
    y = logits_y - K.log(-K.log(U + 1e-20) + 1e-20) # logits + gumbel noise
    y = softmax(K.reshape(y, (-1, N, M)) / tau)
    y = K.reshape(y, (-1, N*M))
    return y

z = Lambda(sampling, output_shape=(M*N,))(logits_y)
generator = Sequential()
generator.add(Dense(256, activation='relu', input_shape=(N*M, )))
generator.add(Dense(512, activation='relu'))
generator.add(Dense(data_dim, activation='sigmoid'))
x_hat = generator(z)

# x_hat = Dense(data_dim, activation='softmax')(Dense(512, activation='relu')(Dense(256, activation='relu')(z)))
def gumbel_loss(x, x_hat):
    q_y = K.reshape(logits_y, (-1, N, M))
    q_y = softmax(q_y)
    log_q_y = K.log(q_y + 1e-20)
    kl_tmp = q_y * (log_q_y - K.log(1.0/M))
    KL = K.sum(kl_tmp, axis=(1, 2))
    elbo = data_dim * bce(x, x_hat) - KL 
    return elbo

vae = Model(x, x_hat)
vae.compile(optimizer='adam', loss=gumbel_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

for e in range(nb_epoch):
    vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=1,
        batch_size=batch_size,
        validation_data=(x_test, x_test))
    K.set_value(tau, np.max([K.get_value(tau) * np.exp(- anneal_rate * e), min_temperature]))