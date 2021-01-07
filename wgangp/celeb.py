try:
    from google.colab import drive

    drive.mount('/content/drive', force_remount=True)
    COLAB = True
    print("Note: using Google CoLab")
#     %tensorflow_version 2.x
except:
    print("Note: not using Google CoLab")
    COLAB = False

if COLAB:
    PROJECT_ROOT = "/content/drive/My Drive/Colab Notebooks/Generative Deep Learning - kuboko"
else:
    PROJECT_ROOT = "../"

LIB_PATH = PROJECT_ROOT

import sys

if not LIB_PATH in sys.path:
    sys.path.append(LIB_PATH)
    print(LIB_PATH + ' has been added to sys.path')

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array
import os, sys
from wgangp.model.generator import Generator
from wgangp.model.critic import Critic
from wgangp.model.wgangp import WGANGP
# from model.discriminator import Discriminator
# from model.gan import Gan

from util import io_utils as io
from util import logger as lgr
from util import config

logger = lgr.get_wgangp_logger("celeb.py")
cfg_exec = config.cfg_wgangp_exec

io.project_root = PROJECT_ROOT
run_folder = io.prepare_wgangp_folders()
logger.info("-------------------- New run of celeb WGANPG. Run folder: %s --------------------", run_folder)
batch_size = cfg_exec['batch_size']

cfg = config.cfg_wgangp
x_train = io.load_celeb_data(cfg,
                             ImageDataGenerator(preprocessing_function=lambda x: (x.astype('float32') - 127.5) / 127.5))

generator = Generator(
    initial_dense_layer_size=(4, 4, 512)
    , upsample=[1, 1, 1, 1]
    , conv_filters=[256, 128, 64, 3]
    , conv_kernel_size=[5, 5, 5, 5]
    , conv_strides=[2, 2, 2, 2]
    , batch_norm_momentum=0.9
    , activation='leaky_relu'
    , dropout_rate=None
    , learning_rate=0.0002
    , z_dim=100
)

gen_model = generator.model
gen_model.summary()

image_size = config.cfg_wgangp_io['image_size']
critic = Critic(
    input_dim=(image_size, image_size, 3)
    , conv_filters=[64, 128, 256, 512]
    , conv_kernel_size=[5, 5, 5, 5]
    , conv_strides=[2, 2, 2, 2]
    , batch_norm_momentum=None
    , activation='leaky_relu'
    , dropout_rate=None
    , learning_rate=0.0002
)

critic_model = critic.model
critic_model.summary()

wgangp = WGANGP(
    generator=generator,
    critic=critic,
    optimiser='adam'
    , grad_weight=10
    , batch_size=batch_size
)

wgangp_critic_model = wgangp.critic_model
wgangp_critic_model.summary()
wgangp_generator_model = wgangp.generator_model
wgangp_generator_model.summary()

print(cfg_exec['mode'], 'mode')
if cfg_exec['mode'] == 'build':
    wgangp.save(run_folder)
else:
    wgangp.load_weights(os.path.join(run_folder, 'weights/weights.h5'))

wgangp.train(
    x_train
    , batch_size=batch_size
    , epochs=cfg_exec['epochs']
    , run_folder=run_folder
    , print_every_n_batches=cfg_exec['print_batches']
    , n_critic=5
    , using_generator=True
)
