import argparse
import torch
from pathlib import Path
import yaml
import glob
from collections import defaultdict
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ITsFlexible.models.egnn_model import flexEGNN


def main(config: dict):
    if config['log']:
        logger = WandbLogger(
            save_dir=Path(config['save_dir']),
            name=f"{config['name']}",
            project=f"{config['logger_params']['project']}",
            group=config['logger_params']['group'],
            config={**config['model_params'], **config['trainer_params'],
                    **config['loader_params'], **config['dataset_params']},
            )
    else:
        logger = None

    model = flexEGNN(
            dataset_config=config['dataset_params'],
            loader_config=config['loader_params'],
            trainer_config=config['trainer_params'],
            run_id=config['name'],
            save_dir=config['save_dir'],
            **config['model_params']
        )

    checkpoint_callback = ModelCheckpoint(
        monitor='pr_auc/val',
        mode='max',
        )

    early_stop_callback = EarlyStopping(
        monitor="pr_auc/val",
        patience=15,
        mode="max"
        )

    trainer = pl.Trainer(
        default_root_dir=config['save_dir'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        **config['trainer_params'])

    # load model from checkpoint
    if config['restore']:
        print("Loading checkpoint & restoring for continued training")
        model = flexEGNN.load_from_checkpoint(
            checkpoint_path=config['restore'],
            dataset_config=config['dataset_params'],
            loader_config=config['loader_params'],
            trainer_config=config['trainer_params'],
            **config['model_params'])

        trainer.resume_from_checkpoint = config['restore']

    trainer.fit(model)
    # save final model parameters
    torch.save({
        'epoch': trainer.current_epoch,
        'model_state_dict': model.state_dict(),
        }, config['save_dir'] + "checkpoint_final.pt")

    if config['test']:
        # load best model
        checkpoint_path = glob.glob(
            config['save_dir'] + '/' +
            config['logger_params']['project'] + '/' +
            '**/checkpoints/*.ckpt')[0]
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

        model.test_set_predictions = []
        trainer.test(model)
        model.save_test_predictions(Path(
            config['save_dir']) / "test_preds.csv"
            )

parser = argparse.ArgumentParser(description='Train CDR3 flexibility model')
parser.add_argument('--predictor', type=str, help='Predictor type', default='loop')

if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Training {args.predictor} model")
    with open(f'../ITsFlexible/trained_model/config_{args.predictor}.yaml') as file_handle:
        config = yaml.safe_load(file_handle)
    config = defaultdict(lambda: None, config)
    config['save_dir'] = (config['save_dir'] + '/' +
                          config['name'] + '/')

    Path(config['save_dir']).mkdir(exist_ok=True, parents=True)

    main(config)
