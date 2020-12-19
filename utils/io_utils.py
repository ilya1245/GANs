import os
import numpy as np
import matplotlib.pyplot as plt

project_root = ''
cfg = []

def prepare_result_folders():

    run_folder = project_root + 'run/{}/'.format(cfg['section'])
    run_folder += '_'.join([cfg['run_id'], cfg['section']])

    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
        # os.mkdir(os.path.join(run_folder, 'viz'))
        os.mkdir(os.path.join(run_folder, 'images'))
        os.mkdir(os.path.join(run_folder, 'weights'))

def load_camel_data():
    data_path = os.path.join(project_root, cfg['data_folder'], cfg['data_file'])
    npy_array = np.load(data_path)
    x_array = npy_array.reshape(npy_array.shape[0], 28, 28, 1)
    plt.imshow(x_array[1].reshape(28, 28), cmap='gray')
    plt.show()

