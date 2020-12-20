from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from tensorflow.keras.models import Model, Sequential

from util import model_utils as mu
from model.generator import Generator
from model.discriminator import Discriminator


def build(generator: Generator, discriminator: Discriminator):
    gen_model = generator.model
    dis_model = discriminator.model
    mu.set_trainable(dis_model, False)
    model_input = Input(shape=(generator.z_dim,), name='adv_input')
    model_output = dis_model(gen_model(model_input))
    model = Model(model_input, model_output)
    model.compile(optimizer=mu.get_opti(generator.optimiser, generator.learning_rate), loss='binary_crossentropy', metrics=['accuracy']
                  , experimental_run_tf_function=False)
    mu.set_trainable(dis_model, True)
    return model