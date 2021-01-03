from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout, Layer

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
import numpy as np

import util.model_utils as mu
from util import io_utils as io

logger = io.get_camel_logger(__name__)


class Decoder():
    def __init__(
            self,
            z_dim,
            conv_filters,
            conv_kernel_size,
            conv_strides,
            shape_before_flattening,
            use_batch_norm=False,
            use_dropout=False
    ):
        self.z_dim = z_dim
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.shape_before_flattening = shape_before_flattening

        self.model = self._build()

    @io.log_method_call(logger)
    def _build(self):

        input = Input(shape=(self.z_dim,), name='input')

        x = Dense(np.prod(self.shape_before_flattening))(input)
        x = Reshape(self.shape_before_flattening)(x)

        for i in range(len(self.conv_filters)):
            conv_t_layer = Conv2DTranspose(
                filters=self.conv_filters[i]
                , kernel_size=self.conv_kernel_size[i]
                , strides=self.conv_strides[i]
                , padding='same'
                , name='conv_t_' + str(i)
            )

            x = conv_t_layer(x)

            if i < len(self.conv_filters) - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        output = x

        self.model = Model(input, output, name='decoder')
