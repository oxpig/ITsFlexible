import pandas as pd
import numpy as np
import os
import yaml
from collections import defaultdict
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as GeoDataLoader
from AbFlex.models.egnn_model import flexEGNN
from AbFlex.base.dataset import LoopGraphDataSet


def classify(infile=None,
             outfile=None,
             predictor='loop',
             config='trained_model/config.yaml',
             weights=None):
    '''Classify antibody CDR and protein loop flexibility.
    
    Args:
        infile (str): Path to input file.
        outfile (str): Path to output file.
        predictor (str): Type of predictor to use (loop or anchors).
        config (str): Path to configuration file.
        weights (str): Path to model weights.
    '''
    script_dir = os.path.dirname(os.path.realpath(__file__))

    config_path = os.path.join(script_dir, config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = defaultdict(lambda: None, config)

    if weights is None:
        if predictor == 'loop':  
            weights = os.path.join(script_dir, 'trained_model/align_loop_top.ckpt')
        elif predictor == 'anchors':
            weights = os.path.join(script_dir, 'trained_model/align_anchors_top.ckpt')
        else:
            raise ValueError("Predictor must be 'loop' or 'anchors'")

    model = flexEGNN.load_from_checkpoint(
                checkpoint_path=weights,
                dataset_config=config['dataset_params'],
                loader_config=config['loader_params'],
                trainer_config=config['trainer_params'],
                **config['model_params'])
    
    ds = LoopGraphDataSet(
            **config['dataset_params']
            )
    ds.populate(input_file=infile)
    loader = GeoDataLoader(ds, batch_size=32, num_workers=4, shuffle=False)

    trainer = pl.Trainer(logger=False)    
    preds = trainer.predict(model=model, dataloaders=loader, return_predictions=True)
    preds = np.concatenate(preds)

    df = pd.read_csv(infile, index_col=0)
    df['preds'] = preds

    if outfile is None:
        return df
    else:
        df.to_csv(outfile)
