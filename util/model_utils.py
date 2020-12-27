from tensorflow.keras.layers import Layer, Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K

class RandomWeightedAverage(Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    """Provides a (random) weighted average between real and generated image samples"""
    def call(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def get_activation_layer(activation):
    if activation == 'leaky_relu':
        layer = LeakyReLU(alpha = 0.2)
    else:
        layer = Activation(activation)
    return layer

def get_opti(optimiser, lr):
    if optimiser == 'adam':
        opti = Adam(lr=lr, beta_1=0.5)
    elif optimiser == 'rmsprop':
        opti = RMSprop(lr=lr)
    else:
        opti = Adam(lr=lr)

    return opti

def set_trainable(model: Model , val):
    model.trainable = val
    for l in model.layers:
        l.trainable = val