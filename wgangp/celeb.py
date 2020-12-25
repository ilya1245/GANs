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
# from model.discriminator import Discriminator
# from model.gan import Gan

from util import io_utils as io
from util import config

logger = io.get_celeb_logger("celeb.py")

io.project_root = PROJECT_ROOT
RUN_FOLDER = io.prepare_celeb_folders()
logger.info("-------------------- New run of celeb WGANPG. Run folder: %s --------------------", RUN_FOLDER)

x_train = io.load_celeb_data()

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

image_size = config.cfg_celeb_io['image_size']
critic = Critic(
    input_dim = (image_size,image_size,3)
    , conv_filters = [64,128,256,512]
    , conv_kernel_size = [5,5,5,5]
    , conv_strides = [2,2,2,2]
    , batch_norm_momentum = None
    , activation = 'leaky_relu'
    , dropout_rate = None
    , learning_rate = 0.0002
)

crt_model = critic.model
crt_model.summary()