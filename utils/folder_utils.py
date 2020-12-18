import os


def prepare_result_folders(project_path, section, run_id):
    RUN_FOLDER = project_path + 'run/{}/'.format(section)
    RUN_FOLDER += '_'.join([run_id, section])

    if not os.path.exists(RUN_FOLDER):
        os.makedirs(RUN_FOLDER)
    # os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))