from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os
import pickle as pkl
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from util import io_utils as io
from util import model_utils as mu
from wgangp.model.generator import Generator
from wgangp.model.critic import Critic

logger = io.get_celeb_logger(__name__)

class WGANGP():
    def __init__(
            self,
            generator: Generator,
            critic: Critic,
            batch_size,
            grad_weight,
            optimiser
    ):
        self.generator = generator
        self.critic = critic
        self.batch_size = batch_size
        self.grad_weight = grad_weight
        self.optimiser = optimiser

        self.name = 'wgangp'
        self.d_losses = []
        self.g_losses = []
        self.epoch = 1

        self.model = self._build()

    @io.log_method_call(logger)
    def _build(self):
        gen_model = self.generator.model
        cirtic_model = self.critic.model

        # Freeze generator's layers while training critic
        mu.set_trainable(gen_model, False)

        # Image input (real sample)
        real_img = Input(shape=self.critic.input_dim)

        # Fake image
        z_disc = Input(shape=(self.generator.z_dim,))
        fake_img = self.generator.model(z_disc)

        # critic determines validity of the real and fake images
        fake = self.critic.model(fake_img)
        valid = self.critic.model(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = mu.RandomWeightedAverage(self.batch_size)([real_img, fake_img])

        # Determine validity of weighted sample
        validity_interpolated = self.critic.model(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'interpolated_samples' argument
        partial_gp_loss = partial(mu.gradient_penalty_loss,
                                  interpolated_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                                  outputs=[valid, fake, validity_interpolated])

        self.critic_model.compile(
            loss=[mu.wasserstein,mu.wasserstein, partial_gp_loss]
            ,optimizer=mu.get_opti(self.optimiser, self.critic.learning_rate)
            ,loss_weights=[1, 1, self.grad_weight]
        )

        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        mu.set_trainable(self.critic.model, False)
        mu.set_trainable(self.generator.model, True)

        # Sampled noise for input to generator
        model_input = Input(shape=(self.generator.z_dim,))
        # Generate images based of noise
        img = self.generator.model(model_input)
        # Discriminator determines validity
        model_output = self.critic.model(img)
        # Defines generator model
        model = Model(model_input, model_output)

        model.compile(optimizer=mu.get_opti(self.optimiser, self.generator.learning_rate)
                           , loss=mu.wasserstein
                           )

        mu.set_trainable(self.critic.model, True)
        return model
