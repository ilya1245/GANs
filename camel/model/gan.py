from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os
import pickle as pkl
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

from util import io_utils as io
from util import model_utils as mu
from generator import Generator
from discriminator import Discriminator

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

    def _build(self):
        logger.debug("%s method is started", self._build.__name__)
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

    def train_generator(self, batch_size):
        logger.debug("%s method is started", self.train_generator.__name__)
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0, 1, (batch_size, self.generator.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train_discriminator(self, x_train, batch_size, using_generator):
        logger.debug("%s method is started", self.train_discriminator.__name__)
        # Create 2 arrays: size=batch_size, one filled by ones, one filled by zeros
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            # Get random batch of true images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        # Generate batch of images using random (normal) input vectors
        noise = np.random.normal(0, 1, (batch_size, self.generator.z_dim))
        # noise = (np.random.randint(0, 2000, (batch_size, self.generator.z_dim)) - 1000)/500
        gen_imgs = self.generator.model.predict(noise)

        d_loss_real, d_acc_real =   self.discriminator.model.train_on_batch(true_imgs, valid)
        d_loss_fake, d_acc_fake =   self.discriminator.model.train_on_batch(gen_imgs, fake)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_acc = (d_acc_real + d_acc_fake) / 2

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=50, using_generator=False):
        logger.debug("%s method is started", self.train.__name__)
        for epoch in range(self.epoch, self.epoch + epochs):

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

    def sample_images(self, run_folder):
        logger.debug("%s method is started", self.sample_images.__name__)
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.generator.z_dim))
        gen_imgs = self.generator.model.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15,15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()

    def save(self, run_folder):
        logger.debug("%s method is started", self.save.__name__)
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

    def save_model(self, run_folder):
        logger.debug("%s method is started", self.save_model.__name__)
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.discriminator.model.save(os.path.join(run_folder, 'discriminator.h5'))
        self.generator.model.save(os.path.join(run_folder, 'generator.h5'))

    def plot_model(self, run_folder):
        logger.debug("%s method is started", self.plot_model.__name__)
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/gan.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.discriminator.model, to_file=os.path.join(run_folder ,'viz/discriminator.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.generator.model, to_file=os.path.join(run_folder ,'viz/generator.png'), show_shapes = True, show_layer_names = True)

    def load_model(self, filepath):
        logger.debug("%s method is started", self.load_model.__name__)
        self.model.load_weights(filepath)