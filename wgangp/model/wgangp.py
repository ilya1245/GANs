from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os
import pickle
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from util import io_utils as io
from util import model_utils as mu
from wgangp.model.generator import Generator
from wgangp.model.critic import Critic

logger = io.get_wgangp_logger(__name__)


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

        self._build_graph()

    @io.log_method_call(logger)
    def _build_graph(self):
        ### Construct Computational Graph for the Critic ###

        # Freeze generator's layers while training critic
        mu.set_trainable(self.generator.model, False)

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
            loss=[mu.wasserstein, mu.wasserstein, partial_gp_loss]
            , optimizer=mu.get_opti(self.optimiser, self.critic.learning_rate)
            , loss_weights=[1, 1, self.grad_weight]
        )

        ### Construct Computational Graph for the Generator ###

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
        self.generator_model = Model(model_input, model_output)

        self.generator_model.compile(
            optimizer=mu.get_opti(self.optimiser, self.generator.learning_rate)
            , loss=mu.wasserstein
        )

        mu.set_trainable(self.critic.model, True)

    @io.log_method_call(logger)
    def train_critic(self, x_train, batch_size, using_generator):

        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = -np.ones((batch_size, 1), dtype=np.float32)
        dummy = np.zeros((batch_size, 1), dtype=np.float32)  # Dummy gt for gradient penalty

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.generator.z_dim))

        d_loss = self.critic_model.train_on_batch([true_imgs, noise], [valid, fake, dummy])
        return d_loss

    @io.log_method_call(logger)
    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(0, 1, (batch_size, self.generator.z_dim))
        return self.generator_model.train_on_batch(noise, valid)

    @io.log_method_call(logger)
    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=10
              , n_critic=5
              , using_generator=False):

        for epoch in range(self.epoch, self.epoch + epochs):

            if epoch % 100 == 0:
                critic_loops = 5
            else:
                critic_loops = n_critic

            for _ in range(critic_loops):
                d_loss = self.train_critic(x_train, batch_size, using_generator)

            g_loss = self.train_generator(batch_size)

            print("%d (%d, %d) [D loss: (%.1f)(R %.1f, F %.1f, G %.1f)] [G loss: %.1f]" % (
            epoch, critic_loops, 1, d_loss[0], d_loss[1], d_loss[2], d_loss[3], g_loss))

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            # If at save interval => save generated image samples
            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.generator_model.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (epoch)))
                self.generator_model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                self.save_model(run_folder)

            self.epoch += 1

    @io.log_method_call(logger)
    def sample_images(self, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.generator.z_dim))
        gen_imgs = self.generator.model.predict(noise)

        # Rescale images 0 - 1

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray_r')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()

    @io.log_method_call(logger)
    def save(self, folder):

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.critic.input_dim
                , self.critic.conv_filters
                , self.critic.conv_kernel_size
                , self.critic.conv_strides
                , self.critic.batch_norm_momentum
                , self.critic.activation
                , self.critic.dropout_rate
                , self.critic.learning_rate
                , self.generator.initial_dense_layer_size
                , self.generator.upsample
                , self.generator.conv_filters
                , self.generator.conv_kernel_size
                , self.generator.conv_strides
                , self.generator.batch_norm_momentum
                , self.generator.activation
                , self.generator.dropout_rate
                , self.generator.learning_rate
                , self.optimiser
                , self.grad_weight
                , self.generator.z_dim
                , self.batch_size
            ], f)

        self.plot_model(folder)

    @io.log_method_call(logger)
    def plot_model(self, run_folder):
        plot_model(self.generator.model, to_file=os.path.join(run_folder, 'viz/generator.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.critic.model, to_file=os.path.join(run_folder, 'viz/critic.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.generator_model, to_file=os.path.join(run_folder, 'viz/generator_model.png'), show_shapes=True,
                   show_layer_names=True)
        # plot_model(self.critic_model, to_file=os.path.join(run_folder, 'viz/critic_model.png'), show_shapes=True,
        #            show_layer_names=True)

    @io.log_method_call(logger)
    def save_model(self, run_folder):
        self.generator.model.save(os.path.join(run_folder, 'generator.h5'))
        self.critic.model.save(os.path.join(run_folder, 'critic.h5'))
        self.generator_model.save(os.path.join(run_folder, 'generator_model.h5'))
        # self.critic_model.save(os.path.join(run_folder, 'critic_model.h5'))

    @io.log_method_call(logger)
    def load_weights(self, filepath):
        self.generator_model.load_weights(filepath)
