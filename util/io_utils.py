import os, sys
import numpy as np
import logging
import config
project_root = None

cfg_exec = config.cfg_exec
cfg_io = config.cfg_io
cfg_log = config.cfg_log

def prepare_run_folders():

    run_folder = project_root + 'run/{}/'.format(cfg_io['section'])
    run_folder += '_'.join([cfg_io['run_id'], cfg_io['data_name']])

    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
        os.mkdir(os.path.join(run_folder, 'viz'))
        os.mkdir(os.path.join(run_folder, 'images'))
        os.mkdir(os.path.join(run_folder, 'weights'))
    return run_folder

def load_camel_data():
    data_path = os.path.join(project_root, cfg_io['data_folder'], cfg_io['data_file'])
    npy_array = np.load(data_path)
    x_array = npy_array.reshape(npy_array.shape[0], 28, 28, 1)
    x = (x_array.astype('float32') - 127.5) / 127.5
    # plt.imshow(x_array[111].reshape(28, 28), cmap='gray')
    # plt.show()
    x = x[:cfg_io['image_quantity']]
    y = [0] * len(x)
    return x, y

def get_camel_logger(module_name):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

    fh = logging.FileHandler(cfg_log['file_name'])
    fh.setLevel(cfg_log['log_level'])
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(cfg_log['console_level'])
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


