import os
import yaml

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    CONFIG_PATH = "/content/drive/My Drive/Colab Notebooks/Generative Deep Learning - kuboko"
except:
    CONFIG_PATH = "../"

with open(os.path.join(CONFIG_PATH, "config.yml"), "r") as ymlfile:
    yml = yaml.load(ymlfile, Loader=yaml.FullLoader)
    _cfg_camel = yml['camel']

cfg_exec = _cfg_camel['exec']
cfg_log = _cfg_camel['logger']
cfg_io = _cfg_camel['io']
