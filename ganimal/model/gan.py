from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os
import pickle as pkl
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

from util import io_utils as io
from util import model_utils as mu
from ganimal.model.generator import Generator
from ganimal.model.discriminator import Discriminator

logger = io.get_camel_logger(__name__)

class Gan():
    def __init__(
            self,
            generator: Generator,
            discriminator: Discriminator
    ):
        self.generator = generator
        self.discriminator = discriminator

        self.name = 'gan'
        self.d_losses = []
        self.g_losses = []
        self.epoch = 1

        self.model = self._build()
        self._compile()

    @io.log_method_call(logger)
    def _build(self):
        gen_model = self.generator.model
        dis_model = self.discriminator.model
        mu.set_trainable(dis_model, False)
        model_input = Input(shape=(self.generator.z_dim,), name='gan_input')
        model_output = dis_model(gen_model(model_input))
        return Model(model_input, model_output)

    def _compile(self):
        dis_model = self.discriminator.model
        self.model.compile(optimizer=mu.get_opti(self.generator.optimiser, self.generator.learning_rate)
                           , loss='binary_crossentropy', metrics=['accuracy']
                           , experimental_run_tf_function=False)
        mu.set_trainable(dis_model, True)

    @io.log_method_call(logger)
    def train_generator(self, batch_size):
        # Train the generator by training the whole GAN with fixed discriminator weights
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.generator.z_dim))
        return self.model.train_on_batch(noise, valid)

    @io.log_method_call(logger)
    def train_discriminator(self, x_train, batch_size, using_generator):

        # Create 2 arrays: size=batch_size, one filled with ones, one filled with zeros
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            # Get a random batch of true images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        # Generate a batch of images using random (normal) input vectors
        noise = np.random.normal(0, 1, (batch_size, self.generator.z_dim))
        # noise = (np.random.randint(0, 2000, (batch_size, self.generator.z_dim)) - 1000)/500
        gen_imgs = self.generator.model.predict(noise)

        d_loss_real, d_acc_real =   self.discriminator.model.train_on_batch(true_imgs, valid)
        # d_metrics_names = self.discriminator.model.metrics_names
        d_loss_fake, d_acc_fake =   self.discriminator.model.train_on_batch(gen_imgs, fake)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_acc = (d_acc_real + d_acc_fake) / 2

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    @io.log_method_call(logger)
    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=50, using_generator=False):
        for epoch in range(self.epoch, self.epoch + epochs):

            # for i in range(0, 10):
            d = self.train_discriminator(x_train, batch_size, using_generator)
            g = self.train_generator(batch_size)

            print("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (
                epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))

            self.d_losses.append(d)
            self.g_losses.append(g)

            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                print("Saving model after %d epochs" % (epoch))
                self.model.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (epoch)))
                self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                self.save_model(run_folder)

            self.epoch += 1

    @io.log_method_call(logger)
    def sample_images(self, run_folder):
        row, col = 3, 3
        noise = np.random.normal(0, 1, (row * col, self.generator.z_dim))
        gen_imgs = self.generator.model.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(row, col, figsize=(15, 15))
        cnt = 0

        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()

    @io.log_method_call(logger)
    def save(self, run_folder):
        with open(os.path.join(run_folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.discriminator.input_dim
                , self.discriminator.conv_filters
                , self.discriminator.conv_kernel_size
                , self.discriminator.conv_strides
                , self.discriminator.batch_norm_momentum
                , self.discriminator.activation
                , self.discriminator.dropout_rate
                , self.discriminator.learning_rate
                , self.discriminator.optimiser
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

        self.plot_model(run_folder)

    @io.log_method_call(logger)
    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.discriminator.model.save(os.path.join(run_folder, 'discriminator.h5'))
        self.generator.model.save(os.path.join(run_folder, 'generator.h5'))

    @io.log_method_call(logger)
    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder, 'viz/gan.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.discriminator.model, to_file=os.path.join(run_folder, 'viz/discriminator.png'),
                   show_shapes=True, show_layer_names=True)
        plot_model(self.generator.model, to_file=os.path.join(run_folder, 'viz/generator.png'), show_shapes=True,
                   show_layer_names=True)

    @io.log_method_call(logger)
    def load_model(self, filepath):
        self.model.load_weights(filepath)