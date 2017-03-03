from keras.layers import initializations
from keras.engine import Layer
from keras import backend as K
import numpy as np
import tensorflow as tf

def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

class GumbelSoftmax(Layer):
    def __init__(self, temperature, hard=False, **kwargs):
        self.supports_masking = True
        self.temperature = temperature
        self.hard = hard
        super(GumbelSoftmax, self).__init__(**kwargs)

    def call(self, x, mask=None):
#         return K.relu(x, alpha=self.alpha)
        y = gumbel_softmax_sample(x, self.temperature)
        if self.hard:
            k = tf.shape(x)[-1]
#             y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
            y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y

    def get_config(self):
        config = {'temperature': self.temperature}
        config = {'hard': self.hard}
        base_config = super(GumbelSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))