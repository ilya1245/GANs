import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array
import zipfile

import logger as lgr
import config

project_root = None
logger = lgr.get_vae_logger(__name__)


@lgr.log_method_call(logger)
def prepare_camel_folders():
    return prepare_folders(config.cfg_camel_io)


@lgr.log_method_call(logger)
def prepare_wgangp_folders():
    return prepare_folders(config.cfg_wgangp_io)


@lgr.log_method_call(logger)
def prepare_vae_folders():
    return prepare_folders(config.cfg_vae_io)


@lgr.log_method_call(logger)
def prepare_folders(cfg_io):
    run_folder = project_root + 'run/{}/'.format(cfg_io['section'])
    run_folder += '_'.join([cfg_io['run_id'], cfg_io['data_name']])

    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
        os.mkdir(os.path.join(run_folder, 'viz'))
        os.mkdir(os.path.join(run_folder, 'images'))
        os.mkdir(os.path.join(run_folder, 'weights'))
    return run_folder


@lgr.log_method_call(logger)
def load_camel_data():
    data_path = os.path.join(project_root, config.cfg_camel_io['data_folder'], config.cfg_camel_io['data_file'])
    npy_array = np.load(data_path)
    x_array = npy_array.reshape(npy_array.shape[0], 28, 28, 1)
    x = (x_array.astype('float32') - 127.5) / 127.5
    # plt.imshow(x_array[111].reshape(28, 28), cmap='gray')
    # plt.show()
    x = x[:config.cfg_camel_io['image_quantity']]
    y = [0] * len(x)
    return x, y


@lgr.log_method_call(logger)
def load_celeb_data(cfg, idg: ImageDataGenerator):
    data_folder = cfg['io']['data_folder']

    if os.path.exists(data_folder):
        x_train = load_images(cfg, idg)
        if x_train.num_classes > 0:
            return x_train

    return load_celeb_data_zip(cfg, idg)


@lgr.log_method_call(logger)
def load_celeb_data_zip(cfg, idg: ImageDataGenerator):
    unzip_folder = cfg['io']['data_folder']
    zip_file = cfg['io']['zip_file']
    image_size = cfg['io']['image_size']

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(unzip_folder)

    return load_images(cfg, idg)


@lgr.log_method_call(logger)
def load_images(cfg, idg: ImageDataGenerator):
    data_folder = cfg['io']['data_folder']
    image_size = cfg['io']['image_size']

    x_train = idg.flow_from_directory(data_folder
                                      , target_size=(image_size, image_size)
                                      , batch_size=cfg['exec']['batch_size']
                                      # , shuffle=True
                                      , class_mode='input'
                                      , subset="training"
                                      )
    # plt.imshow(x_train[0][0][0])
    # plt.show()
    return x_train
