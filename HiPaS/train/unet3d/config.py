import argparse
import torch
import yaml
from HiPaS.train.unet3d import utils

logger = utils.get_logger('ConfigLoader')


def load_config():
    # parser = argparse.ArgumentParser(description='UNet3D')
    # parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    # args = parser.parse_args()
    config = yaml.safe_load(open("/home/chuy/PythonProjects/HiPaS_AV_Segmentation-main/HiPaS/train_config_segmentation.yaml",
                                 'r'))

    device = config.get('device', None)
    if device == 'cpu':
        logger.warning('CPU mode forced in config, this will likely result in slow training/prediction')
        config['device'] = 'cpu'
        return config

    if torch.cuda.is_available():
        config['device'] = 'cuda'
    else:
        logger.warning('CUDA not available, using CPU')
        config['device'] = 'cpu'
    return config


def _load_config_yaml():
    return yaml.safe_load(open("/home/chuy/PythonProjects/3dunet/train_config_segmentation.yaml",
                               'r'))
