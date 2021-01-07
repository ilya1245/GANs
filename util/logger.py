import sys

import logging
import config


def get_camel_logger(module_name):
    return get_logger(module_name, config.cfg_camel_log)


def get_wgangp_logger(module_name):
    return get_logger(module_name, config.cfg_wgangp_log)


def get_vae_logger(module_name):
    return get_logger(module_name, config.cfg_vae_log)


def get_logger(module_name, cfg_log):
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


def log_method_call(logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug("%s method is started", func.__name__)
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator
