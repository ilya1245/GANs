from tensorflow.keras.layers import Input, Conv2D, Dense, Conv2DTranspose, Reshape, Activation, \
    BatchNormalization, Dropout, UpSampling2D

from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

from util import logger as lgr
import numpy as np
import util.model_utils as mu

logger = lgr.get_wgangp_logger(__name__)


class Generator():
    def __init__(
            self
            , initial_dense_layer_size
            , upsample
            , conv_filters
            , conv_kernel_size
            , conv_strides
            , batch_norm_momentum
            , activation
            , dropout_rate
            , learning_rate
            , z_dim
    ):
        self.initial_dense_layer_size = initial_dense_layer_size
        self.upsample = upsample
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.batch_norm_momentum = batch_norm_momentum
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.z_dim = z_dim

        self.name = 'generator'
        self.weight_init = RandomNormal(mean=0., stddev=0.02) # 'he_normal' #RandomNormal(mean=0., stddev=0.02)
        # self.losses = []

        self.model = self._build()

    @lgr.log_method_call(logger)
    def _build(self):
        input = Input(shape=(self.z_dim,), name='input')

        x = Dense(np.prod(self.initial_dense_layer_size), kernel_initializer = self.weight_init)(input)
        if self.batch_norm_momentum:
            x = BatchNormalization(momentum = self.batch_norm_momentum)(x)

        x = mu.get_activation_layer(self.activation)(x)

        x = Reshape(self.initial_dense_layer_size)(x)

        if self.dropout_rate:
            x = Dropout(rate = self.dropout_rate)(x)

        x = self._append_conv_layers(x)

        return Model(input, x)

    def _append_conv_layers(self, x):
        for i in range(len(self.conv_filters)):

            if self.upsample[i] == 2:
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters = self.conv_filters[i]
                    , kernel_size = self.conv_kernel_size[i]
                    , padding = 'same'
                    , name = 'conv_' + str(i)
                    , kernel_initializer = self.weight_init
                )(x)
            else:

                x = Conv2DTranspose(
                    filters = self.conv_filters[i]
                    , kernel_size = self.conv_kernel_size[i]
                    , padding = 'same'
                    , strides = self.conv_strides[i]
                    , name = 'conv_' + str(i)
                    , kernel_initializer = self.weight_init
                )(x)

            if i < len(self.conv_filters) - 1:

                if self.batch_norm_momentum:
                    x = BatchNormalization(momentum = self.batch_norm_momentum)(x)

                x = mu.get_activation_layer(self.activation)(x)

            else:
                x = Activation('tanh')(x)

        return x