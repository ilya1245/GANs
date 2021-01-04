from tensorflow.keras.models import Model
import tensorflow as tf

from util import io_utils as io

logger = io.get_vae_logger(__name__)

class VaeModel(Model):
    def __init__(self, encoder, decoder, r_loss_factor, **kwargs):
        super(VaeModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.r_loss_factor = r_loss_factor

    @io.log_method_call(logger)
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.square(data - reconstruction), axis=[1, 2, 3]
            )
            reconstruction_loss *= self.r_loss_factor
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_sum(kl_loss, axis=1)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    @io.log_method_call(logger)
    def call(self, inputs):
        latent = self.encoder(inputs)
        return self.decoder(latent)