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