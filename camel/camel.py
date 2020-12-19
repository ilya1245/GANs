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
from utils import io_utils as io

with open(os.path.join(PROJECT_ROOT, "config.yml"), "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

exec = cfg['exec']
io.cfg = cfg['io']
io.project_root = PROJECT_ROOT
a = io.prepare_run_folders()
io.load_camel_data()

# data_path = os.path.join(PROJECT_ROOT, io.cfg['data_folder'], io.cfg['data_file'])
