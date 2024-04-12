import random
import torch
from HiPaS.train.unet3d.config import load_config
from HiPaS.train.unet3d.trainer import create_trainer


def main():
    # Load and log experiment configuration
    config = load_config()
    trainer = create_trainer(config)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
