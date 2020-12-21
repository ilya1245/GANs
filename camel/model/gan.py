from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from tensorflow.keras.models import Model, Sequential
import os
import pickle as pkl

from util import model_utils as mu
from generator import Generator
from discriminator import Discriminator

class Gan():
    def __init__(
            self,
            generator: Generator,
            discriminator: Discriminator
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.name = 'gan'
        self.model = self._build()
        self._compile()

    def _build(self):
        gen_model = self.generator.model
        dis_model = self.discriminator.model
        mu.set_trainable(dis_model, False)
        model_input = Input(shape=(self.generator.z_dim,), name='gan_input')
        model_output = dis_model(gen_model(model_input))
        return  Model(model_input, model_output)

    def _compile(self):
        dis_model = self.discriminator.model
        self.model.compile(optimizer=mu.get_opti(self.generator.optimiser, self.generator.learning_rate)
                           , loss='binary_crossentropy', metrics=['accuracy']
                           , experimental_run_tf_function=False)
        mu.set_trainable(dis_model, True)

    def save(self, folder):
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim
                , self.discriminator.conv_filters
                , self.discriminator.conv_kernel_size
                , self.discriminator.conv_strides
                , self.discriminator.batch_norm_momentum
                , self.discriminator.activation
                , self.discriminator.dropout_rate
                , self.discriminator.learning_rate
                , self.generator.initial_dense_layer_size
                , self.generator.upsample
                , self.generator.conv_filters
                , self.generator.conv_kernel_size
                , self.generator.conv_strides
                , self.generator.batch_norm_momentum
                , self.generator.activation
                , self.generator.dropout_rate
                , self.generator.learning_rate
                , self.generator.optimiser
                , self.generator.z_dim
            ], f)

        self.plot_model(folder)

    def load_model(self, filepath):
        self.model.load_weights(filepath)