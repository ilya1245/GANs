from tensorflow.keras.layers import Input, Flatten, Conv2D, Dense, Conv2DTranspose, Reshape, Activation, \
    BatchNormalization, Dropout, UpSampling2D

from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

from util import logger as lgr
import numpy as np
import util.model_utils as mu

logger = lgr.get_wgangp_logger(__name__)


class Critic():
    def __init__(
            self
            , input_dim
            , conv_filters
            , conv_kernel_size
            , conv_strides
            , batch_norm_momentum
            , activation
            , dropout_rate
            , learning_rate
    ):

        self.input_dim = input_dim
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.batch_norm_momentum = batch_norm_momentum
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.weight_init = RandomNormal(mean=0., stddev=0.02)  #  'he_normal' #RandomNormal(mean=0., stddev=0.02)

        self.model = self._build()

    @lgr.log_method_call(logger)
    def _build(self):

        input = Input(shape=self.input_dim, name='input')

        x = input

        for i in range(len(self.conv_filters)):

            x = Conv2D(
                filters=self.conv_filters[i]
                , kernel_size=self.conv_kernel_size[i]
                , strides=self.conv_strides[i]
                , padding='same'
                , name='conv_' + str(i)
                , kernel_initializer=self.weight_init
            )(x)

            if self.batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum=self.batch_norm_momentum)(x)

            x = mu.get_activation_layer(self.activation)(x)

            if self.dropout_rate:
                x = Dropout(rate=self.dropout_rate)(x)

        x = Flatten()(x)

        # x = Dense(512, kernel_initializer = self.weight_init)(x)

        # x = self.get_activation(self.activation)(x)

        output = Dense(1, activation=None
                       , kernel_initializer=self.weight_init
                       )(x)

        return Model(input, output)
