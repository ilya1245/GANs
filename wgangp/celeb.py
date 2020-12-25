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

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array
import os, sys

from util import io_utils as io
from util import config

logger = io.get_celeb_logger("celeb.py")
cfg_exec = config.cfg_celeb_exec

io.project_root = PROJECT_ROOT
RUN_FOLDER = io.prepare_celeb_folders()

x_train = io.load_celeb_data()

