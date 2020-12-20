from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from tensorflow.keras.optimizers import Adam, RMSprop


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