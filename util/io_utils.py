import os
import numpy as np
import matplotlib.pyplot as plt

project_root = None
cfg = []

def prepare_run_folders():

    run_folder = project_root + 'run/{}/'.format(cfg['section'])
    run_folder += '_'.join([cfg['run_id'], cfg['data_name']])

    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
        os.mkdir(os.path.join(run_folder, 'viz'))
        os.mkdir(os.path.join(run_folder, 'images'))
        os.mkdir(os.path.join(run_folder, 'weights'))
    return run_folder

def load_camel_data():
    data_path = os.path.join(project_root, cfg['data_folder'], cfg['data_file'])
    npy_array = np.load(data_path)
    x_array = npy_array.reshape(npy_array.shape[0], 28, 28, 1)
    x = (x_array.astype('float32') - 127.5) / 127.5
    # plt.imshow(x_array[111].reshape(28, 28), cmap='gray')
    # plt.show()
    x = x[:cfg['image_quantity']]
    y = [0] * len(x)
    return x, y


