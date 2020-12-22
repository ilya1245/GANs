from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, Dropout

from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

import util.model_utils as mu
from util import io_utils as io

logger = io.get_camel_logger(__name__)

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
        logger.debug("%s method is started", self._build.__name__)
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

            x = mu.get_activation_layer(self.activation)(x)

            if self.dropout_rate:
                x = Dropout(rate=self.dropout_rate)(x)

        x = Flatten()(x)

        x = Dense(1, activation='sigmoid', kernel_initializer=self.weight_init)(x)

        return Model(input, x)

    def _compile(self):
        self.model.compile(
            optimizer=mu.get_opti(self.optimiser, self.learning_rate)
            , loss='binary_crossentropy'
            , metrics=['accuracy']
        )
