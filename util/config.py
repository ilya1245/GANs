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
    cfg_camel = yml['camel']
    cfg_wgangp = yml['wgangp']
    cfg_vae = yml['vae']

cfg_camel_exec = cfg_camel['exec']
cfg_camel_log = cfg_camel['logger']
cfg_camel_io = cfg_camel['io']

cfg_wgangp_exec = cfg_wgangp['exec']
cfg_wgangp_log = cfg_wgangp['logger']
cfg_wgangp_io = cfg_wgangp['io']

cfg_vae_exec = cfg_vae['exec']
cfg_vae_log = cfg_vae['logger']
cfg_vae_io = cfg_vae['io']
