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
    _cfg_wgangp = yml['wgangp']
    _cfg_vae = yml['vae']

cfg_camel_exec = _cfg_camel['exec']
cfg_camel_log = _cfg_camel['logger']
cfg_camel_io = _cfg_camel['io']

cfg_wgangp_exec = _cfg_wgangp['exec']
cfg_wgangp_log = _cfg_wgangp['logger']
cfg_wgangp_io = _cfg_wgangp['io']

cfg_vae_exec = _cfg_vae['exec']
cfg_vae_log = _cfg_vae['logger']
cfg_vae_io = _cfg_vae['io']


