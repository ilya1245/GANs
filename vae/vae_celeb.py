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

from vae.model.encoder import Encoder
from vae.model.decoder import Decoder
from vae.model.vae import VAE

from util import io_utils as io
from util import config

logger = io.get_vae_logger("vae_celeb.py")
cfg_exec = config.cfg_vae_exec

io.project_root = PROJECT_ROOT
run_folder = io.prepare_vae_folders()
logger.info("-------------------- New run of celeb WGANPG. Run folder: %s --------------------", run_folder)

input_dim = (128, 128, 3)

batch_size = cfg_exec['batch_size']

x_train = io.load_celeb_data_zip()

encoder = Encoder(input_dim,
                  conv_filters=[32, 64, 64, 64],
                  conv_kernel_size=[3, 3, 3, 3],
                  conv_strides=[2, 2, 2, 2],
                  z_dim=200,
                  use_batch_norm=True,
                  use_dropout=True)

decoder = Decoder(z_dim=200,
                  conv_filters=[32, 64, 64, 64],
                  conv_kernel_size=[3, 3, 3, 3],
                  conv_strides=[2, 2, 2, 2],
                  shape_before_flattening=encoder.shape_before_flattening,
                  use_batch_norm=True,
                  use_dropout=True)

vae = VAE(encoder=encoder,
          decoder=decoder,
          r_loss_factor=10000)

vae.compile(learning_rate=0.0005)
