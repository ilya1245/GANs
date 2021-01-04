from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

import os
import pickle

from util import io_utils as io
from vae.model.callback import CustomCallback, step_decay_schedule
from vae.model.vae_model import VaeModel

logger = io.get_vae_logger(__name__)

class VAE():
    def __init__(self,
                 encoder,
                 decoder,
                 r_loss_factor
                 ):

        self.name = 'variational_autoencoder'

        self.encoder = encoder
        self.decoder = decoder
        self.r_loss_factor = r_loss_factor

        self._build()

    @io.log_method_call(logger)
    def _build(self):
        self.model = VaeModel(self.encoder, self.decoder, self.r_loss_factor)

    @io.log_method_call(logger)
    def compile(self, learning_rate):
        self.learning_rate = learning_rate
        self.model.compile(optimizer=Adam(lr=learning_rate))

    @io.log_method_call(logger)
    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.encoder.input_dim
                , self.encoder.conv_filters
                , self.encoder.conv_kernel_size
                , self.encoder.conv_strides
                , self.decoder.conv_t_filters
                , self.decoder.conv_t_kernel_size
                , self.decoder.conv_t_strides
                , self.decoder.z_dim
                , self.decoder.use_batch_norm
                , self.decoder.use_dropout
            ], f)

        self.plot_model(folder)

    @io.log_method_call(logger)
    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    @io.log_method_call(logger)
    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0, lr_decay=1):

        checkpoint_filepath = os.path.join(run_folder, "training/cp.ckpt")
        checkpoint = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)

        self.model.fit(
            x_train
            , x_train
            , batch_size=batch_size
            , shuffle=True
            , epochs=epochs
            , initial_epoch=initial_epoch
            , callbacks=[checkpoint]
        )

    @io.log_method_call(logger)
    def train_with_generator(self, data_flow, epochs, steps_per_epoch, run_folder, print_every_n_batches=100,
                             initial_epoch=0, lr_decay=1, ):

        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath = os.path.join(run_folder, "training/cp.ckpt")
        checkpoint = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)

        callbacks_list = [checkpoint, custom_callback, lr_sched]

        self.model.save_weights(checkpoint_filepath)

        self.model.fit(
            data_flow
            , shuffle=True
            , epochs=epochs
            , initial_epoch=initial_epoch
            , callbacks=callbacks_list
            , steps_per_epoch=steps_per_epoch
        )

    @io.log_method_call(logger)
    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.encoder, to_file=os.path.join(run_folder, 'viz/encoder.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.decoder, to_file=os.path.join(run_folder, 'viz/decoder.png'), show_shapes=True,
                   show_layer_names=True)
