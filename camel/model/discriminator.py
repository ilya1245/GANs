from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomNormal

import numpy as np
import util.model_utils as gu


class Discriminator():
    def __init__(
            self,
            input_dim,
            conv_filters,
            conv_kernel_size,
            conv_strides,
            batch_norm_momentum,
            activation,
            dropout_rate,
            learning_rate,
            optimiser
    ):
        self.input_dim = input_dim
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.batch_norm_momentum = batch_norm_momentum
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimiser = optimiser

        self.name = 'discriminator'
        self.weight_init = RandomNormal(mean=0., stddev=0.02)
        self.losses = []

        self.model = self._build()
        self._compile()

    def _build(self):
        input = Input(shape=self.input_dim)
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

            x = gu.get_activation_layer(self.activation)(x)

            if self.dropout_rate:
                x = Dropout(rate=self.dropout_rate)(x)

        x = Flatten()(x)

        x = Dense(1, activation='sigmoid', kernel_initializer=self.weight_init)(x)

        return Model(input, x)

    def _compile(self):
        self.model.compile(
            optimizer=gu.get_opti(self.optimiser, self.learning_rate)
            , loss='binary_crossentropy'
            , metrics=['accuracy']
        )
