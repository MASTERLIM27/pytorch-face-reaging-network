import argparse
from pathlib import Path
import warnings

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.augmentations import transform_normalize
from models.descriminator import PatchGANDiscriminator
from models.generator import Generator
from datasets.fran_dataset import FRANDataset
from training.trainer import FRAN
import os
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

data_dir = Path('./datasets/aged_synthetic_dataset/')

def get_args():
    parser = argparse.ArgumentParser(description='Train FRAN model.')
    parser.add_argument('--data_dir', '-C', type=str, default=data_dir, help='directory for data')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    image_meta = pd.read_csv(args.data_dir / "image_meta.csv")
    image_meta = image_meta.drop(columns=['Unnamed: 0'])

    train_dataset = FRANDataset(image_meta, transform_normalize, args.data_dir / "synthetic_images")
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    logger = TensorBoardLogger("logs", name="model")
    csv_logger = CSVLogger(save_dir="logs", name="loss")

    fran_model = FRAN(Generator(), PatchGANDiscriminator())

    fran_trainer = pl.Trainer(
        precision='16-mixed',
        devices=1,
        max_epochs=20,
        logger=[logger, csv_logger],
        callbacks =[pl.callbacks.ModelCheckpoint(
            every_n_train_steps=500*2,
            dirpath=os.path.join(logger.log_dir, "model_checkpoints"),
            filename='fran-{step:05d}',
            save_last=True,
            save_top_k=-1
        )]
    ) 

    fran_trainer.fit(fran_model, dataloader)