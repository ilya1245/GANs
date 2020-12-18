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

# import os
import matplotlib.pyplot as plt
from utils import folder_utils as fu

# run params
SECTION = 'camel'
RUN_ID = '005'

fu.prepare_result_folders(PROJECT_PATH, SECTION, RUN_ID)
