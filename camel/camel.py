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
    PROJECT_PATH = "/content/drive/My Drive/Colab Notebooks/Generative Deep Learning - kuboko/"
    LIB_PATH = PROJECT_PATH
else:
    PROJECT_PATH = "../"
    LIB_PATH = "../"

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

with open(PROJECT_PATH + "config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# print(cfg["exec"]["section"])
# run params
exec = cfg['exec']
static = cfg['static']


io.prepare_result_folders(PROJECT_PATH, exec['section'], exec['run_id'])

mypath = os.path.join(PROJECT_PATH, exec['data_folder'])
filenames = np.array(glob(os.path.join(mypath, '*.*')))
print(mypath)
print(filenames)