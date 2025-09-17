'''Script to predict CDR3 flexibility using the ITsFlexible models'''

import argparse
import pandas as pd
import numpy as np
import yaml
from collections import defaultdict
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as GeoDataLoader
import torch
from ITsFlexible.models.egnn_model import flexEGNN
from ITsFlexible.base.dataset import LoopGraphDataSet


def main(dataset_path, predictor, accelerator, batch_size):
    config_path = f'../ITsFlexible/trained_model/config_{predictor}.yaml'
    checkpoint_path = f'../ITsFlexible/trained_model/align_{predictor}_top.ckpt'

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = defaultdict(lambda: None, config)

    model = flexEGNN.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                dataset_config=config['dataset_params'],
                loader_config=config['loader_params'],
                trainer_config=config['trainer_params'],
                **config['model_params'])

    ds = LoopGraphDataSet(
            **config['dataset_params'],
            predict=True,
    )
    ds.populate(input_file=dataset_path)
    loader = GeoDataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False)

    if (accelerator == 'auto' and not torch.cuda.is_available()) or accelerator == 'cpu':
        trainer = pl.Trainer(logger=False, accelerator='cpu')
    else:
        # Force devices=1 to use only one GPU/device
        trainer = pl.Trainer(logger=False, accelerator=accelerator, devices=1)
    
    preds = trainer.predict(
        model=model, dataloaders=loader, return_predictions=True
        )
    preds = np.concatenate(preds)

    df = pd.read_csv(dataset_path, index_col=0)
    df['preds'] = preds
    df.to_csv('predictions.csv')


parser = argparse.ArgumentParser(description='Predict CDR3 flexibility')
parser.add_argument('--dataset', type=str, help='Path to dataset', default='../data/CDRH3_test_align_loop.csv')
parser.add_argument('--predictor', type=str, help='Predictor type', default='loop')
parser.add_argument('--accelerator', type=str, help='Accelerator type', default='auto')
parser.add_argument('--batch_size', type=int, help='Batch size for prediction', default=32)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.dataset, args.predictor, args.accelerator, args.batch_size)
