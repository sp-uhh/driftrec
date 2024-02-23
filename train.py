import argparse
from argparse import ArgumentParser
import logging

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from driftrec.backbones.shared import BackboneRegistry
from driftrec.data_modules import DataModuleRegistry
from driftrec.sdes import SDERegistry
from driftrec.model import ModelRegistry
from driftrec.util.params import get_argparse_groups


if __name__ == '__main__':
     txtlogger = logging.getLogger("driftrec")
     txtlogger.setLevel(logging.INFO)

     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--model", type=str, choices=ModelRegistry.get_all_names(), default="ScoreModel")
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--data_module", type=str, choices=DataModuleRegistry.get_all_names(), default="imagefolder")
          parser_.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B, using local default logger instead")
          parser_.add_argument("--discriminatively", action="store_true", help="Train the backbone as a discriminative model (regression baseline) instead")

     temp_args, _ = base_parser.parse_known_args()
     # Get the classes from the registries by name
     model_name = temp_args.model
     backbone_name = temp_args.backbone
     sde_name = temp_args.sde
     model_cls = ModelRegistry.get_by_name(model_name)
     data_module_name = temp_args.data_module
     backbone_cls = BackboneRegistry.get_by_name(backbone_name)
     sde_cls = SDERegistry.get_by_name(sde_name)
     data_module_cls = DataModuleRegistry.get_by_name(data_module_name)
     # Add specific args for pl.Trainer, Model, SDE, Backbone and DataModule
     pl.Trainer.add_argparse_args(parser)
     model_cls.add_argparse_args(
          parser.add_argument_group("Model", description=model_cls.__name__))
     sde_cls.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_cls.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # sadly PL has its own add_argparse_args style for DataModules, so we're not adding our own argument group here
     data_module_cls.add_argparse_args(parser)
     # Parse args and separate into groups
     parser.set_defaults(max_epochs=300)
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser, args)

     # Gather set of keys to not pass onto the model class, since we process them here (in train.py) directly
     ignored_keys = set([
          *arg_groups["pl.Trainer"].keys(),
          "model", "backbone", "sde", "data_module", "no_wandb", "discriminatively"
     ])
     # Initialize model
     model = model_cls(backbone_name=backbone_name, sde_name=sde_name, data_module_name=data_module_name, **{
          k: v for k, v in vars(args).items()
          if k not in ignored_keys
     })

     # Set up logger configuration
     if args.no_wandb:
          logger = TensorBoardLogger(save_dir="logs", name="tensorboard")
     else:
          logger = WandbLogger(project="driftrec", log_model=True, save_dir="logs")
          logger.experiment.log_code(".")

     # Set up callbacks for logger
     callbacks = [ModelCheckpoint(dirpath=f"logs/{logger.version}", save_last=True, filename='{epoch}-last')]
     valid_loss_callback = ModelCheckpoint(
          dirpath=f"logs/{logger.version}",
          save_top_k=1, monitor="valid_loss", mode="min", filename="{epoch}-{valid_loss:.2f}")
     callbacks += [valid_loss_callback]

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer.from_argparse_args(
          argparse.Namespace(**arg_groups["pl.Trainer"]), strategy=DDPStrategy(find_unused_parameters=False),
          logger=logger, log_every_n_steps=10, callbacks=callbacks
     )
     # A bit of a hack to force construction of the datasets before `trainer.fit`
     model.data_module.setup(stage="fit")
     txtlogger.info(f"Training with {len(model.data_module.ds_train)} data points")

     # Train model
     trainer.fit(model)
