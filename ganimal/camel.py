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

import os
from glob import glob
import numpy as np
# import matplotlib.pyplot as plt
import yaml
from util import io_utils as io
from util import logger as lgr
from util import config
from ganimal.model.generator import Generator
from ganimal.model.discriminator import Discriminator
from ganimal.model.gan import Gan

logger = lgr.get_camel_logger("cammel.py")
cfg_exec = config.cfg_camel_exec

io.project_root = PROJECT_ROOT
RUN_FOLDER = io.prepare_camel_folders()
logger.info("-------------------- New run of camel GAN. Run folder: %s --------------------", RUN_FOLDER)

(x_train, y_train) = io.load_camel_data()

generator = Generator(
    initial_dense_layer_size=(7, 7, 64)
    , upsample=[2, 2, 1, 1]
    , conv_filters=[128, 64, 64, 1]
    , conv_kernel_size=[5, 5, 5, 5]
    , conv_strides=[1, 1, 1, 1]
    , batch_norm_momentum=0.9
    , activation='relu'
    , dropout_rate=None
    , learning_rate=0.0004
    , optimiser='rmsprop'
    , z_dim=100)

gen_model = generator.model
gen_model.summary()

discriminator = Discriminator(
    input_dim=(28, 28, 1)
    , conv_filters=[64, 64, 128, 128]
    , conv_kernel_size=[5, 5, 5, 5]
    , conv_strides=[2, 2, 2, 1]
    , batch_norm_momentum=None
    , activation='relu'
    , dropout_rate=0.4
    , learning_rate=0.0008
    , optimiser='rmsprop'
)

dis_model = discriminator.model
dis_model.summary()

gan = Gan(
    generator=generator,
    discriminator=discriminator
)

gan_model = gan.model
gan_model.summary()

print(cfg_exec['mode'], 'mode')
if cfg_exec['mode'] == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

gan.train(
    x_train
    , batch_size=cfg_exec['batch_size']
    , epochs=cfg_exec['epochs']
    , run_folder=RUN_FOLDER
    , print_every_n_batches=cfg_exec['print_batches']
)