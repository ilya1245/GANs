from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout, Layer

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal

import util.model_utils as mu
from util import io_utils as io

logger = io.get_camel_logger(__name__)

class Encoder():
    def __init__(
            self
            , input_dim
            , conv_filters
            , conv_kernel_size
            , conv_strides
            , z_dim
            , use_batch_norm=False
            , use_dropout=False
    ):
        self.input_dim = input_dim
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.z_dim = z_dim

        self.model = self._build()

    @io.log_method_call(logger)
    def _build(self):
        input = Input(shape=self.input_dim, name='input')

        x = input

        for i in range(len(self.conv_filters)):
            conv_layer = Conv2D(
                filters=self.conv_filters[i]
                , kernel_size=self.conv_kernel_size[i]
                , strides=self.conv_strides[i]
                , padding='same'
                , name='conv_' + str(i)
            )

            x = conv_layer(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        self.shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        self.z = mu.Sampling(name='output')([self.mu, self.log_var])

        self.model = Model(input, [self.mu, self.log_var, self.z], name='encoder')
