import abc
import math
import logging

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
from torchvision.utils import make_grid

import wandb

from driftrec import sampling
from driftrec.sdes import SDERegistry
from driftrec.backbones import BackboneRegistry
from driftrec.data_modules import DataModuleRegistry
from driftrec.util.registry import Registry


logger = logging.getLogger("driftrec")
ModelRegistry = Registry("Model")


class Model(pl.LightningModule, abc.ABC):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        parser.add_argument("--std_weighting_exponent", type=float, default=0, help="Weight the losses by `std**exponent`. 0 by default, i.e., no weighting applied")
        parser.add_argument("--optimizer", type=str, default="adam", choices=("adam", "adamw", "sgd"), help="The optimizer to use.")
        return parser

    @abc.abstractmethod
    def _init_backbone(self, backbone_name, **init_kwargs):
        pass

    def __init__(
        self, backbone_name, data_module_name, lr=1e-4, optimizer='adam', loss_type='mse', ema_decay=0.999,
        std_weighting_exponent=0.0,
        **kwargs
    ):
        """
        Create a new Model.

        Args:
            backbone_name: Name of a registered Backbone DNN class.
            data_module_name: Name of a registered DataModule class.
            optimizer: The name of the optimizer to use (Options: adam, adamw, sgd)
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            loss_type: The type of loss to use. Options are 'mse' (default), 'mae'
        """
        super().__init__()

        # Initialize data module
        data_module_cls = DataModuleRegistry.get_by_name(data_module_name)
        self.data_module = data_module_cls(**kwargs)
        self.num_channels = self.data_module.get_num_channels()

        # Initialize Backbone DNN
        self.backbone = self._init_backbone(backbone_name, **kwargs)

        # Store hyperparams
        self.optimizer = optimizer
        self.lr = lr
        self.loss_type = loss_type
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self.std_weighting_exponent = std_weighting_exponent

    def _finalize_init(self, *ignored_args, **ignored_kwargs):
        # Save hyperparameters after all else is done
        self.save_hyperparameters(ignore=['no_wandb'])

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)


    # ===== Training / Validation steps =====

    @abc.abstractmethod
    def _step(batch, batch_idx):
        pass

    @abc.abstractmethod
    def _log_evaluation(self):
        pass

    def _log_reconstruction(self, x, xhat, y):
        showns = (x, xhat, y)
        interleaved = torch.flatten(torch.stack(showns, dim=1), start_dim=0, end_dim=1)
        img = make_grid(interleaved.cpu(), nrow=3)
        self.logger.experiment.log({"reconstruction": wandb.Image(img, caption="Left: x0, Middle: xhat, Right: y")})

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, mode='train')
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, mode='val')
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        # Evaluate performance with some task-oriented metrics / log sampled images
        if batch_idx == 0:
            self._log_evaluation(batch, batch_idx)
        return loss

    def _get_loss(self, err, std):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()

        if self.std_weighting_exponent is not None and self.std_weighting_exponent != 0:
            weighting = std**self.std_weighting_exponent
        else:
            weighting = 1

        losses = weighting * losses
        avg_loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return avg_loss


    # ===== Dataloader =====

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()


    # ===== Optimizer =====

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Update the EMA params after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())


    # ===== EMA handling overrides =====

    def on_load_checkpoint(self, checkpoint):
        self.ema.load_state_dict(checkpoint['ema'])

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if mode == False and not no_ema:
            # eval
            self.ema.store(self.parameters())        # store current params in EMA
            self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
        else:
            # train
            if self.ema.collected_params is not None:
                self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def to(self, *args, **kwargs):
        # Override PyTorch .to() to also transfer the EMA of the model weights
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)


@ModelRegistry.register("ScoreModel")
class ScoreModel(Model):
    @staticmethod
    def add_argparse_args(parser):
        parser = Model.add_argparse_args(parser)
        parser.add_argument("--t-eps", type=float, default=0.01, help="Minimum process time. 0.01 by default.")
        return parser

    def _init_backbone(self, backbone_name, **init_kwargs):
        backbone_cls = BackboneRegistry.get_by_name(backbone_name)
        # use 2*num_channels to input torch.cat((x_t, y), dim=1)
        return backbone_cls(in_channels=2*self.num_channels, out_channels=self.num_channels, **init_kwargs)

    def __init__(self, *args, sde_name, t_eps, **kwargs):
        """
        Create a new ScoreModel.

        Args:
            backbone_name: Name of a registered Backbone DNN class.
            data_module_name: Name of a registered DataModule class.
            optimizer: The name of the optimizer to use (Options: adam, adamw, sgd)
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            loss_type: The type of loss to use. Options are 'mse' (default), 'mae'

            sde_name: The SDE that defines the diffusion process.
            t_eps: The minimum time to practically run for to avoid issues very close to zero.
        """
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde_name)
        self.sde = sde_cls(**kwargs)
        logger.info(f"Using SDE: {self.sde}")
        self.t_eps = t_eps

        # Call super's __init__ only here, after special setup has already happened
        super().__init__(*args, **kwargs)
        # Finalize init
        self._finalize_init(**kwargs)

    def forward(self, x, t, y):
        # Determine sigmas from t
        sigmas = self.sde._std(t)
        # Concatenate y as an extra channel
        dnn_input = torch.cat((x, y), dim=1)
        score = self.backbone(dnn_input, time=t, sigmas=sigmas)
        return score

    def _random_forward_state(self, x0, y):
        t = torch.rand(x0.shape[0], device=x0.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean = self.sde._mean(x0, t, y)
        std = self.sde._std(t)
        z = torch.randn_like(x0)
        return mean, std, z, t

    def _step(self, batch, batch_idx, mode=None):
        x0, y = batch
        mu_t, std_t, z, t = self._random_forward_state(x0, y)
        sigmas = std_t[:,None,None,None]
        xt = mu_t + sigmas*z

        score = self(xt, t, y)
        error = score*sigmas + z  # = (equiv. to)  score + z/sigmas  =  score - (-z/sigmas)

        loss = self._get_loss(error, sigmas)
        return loss

    def _log_evaluation(self, batch, batch_idx):
        x, y = batch
        sampler = self.get_pc_sampler('reverse_diffusion', 'none', y=y, N=100, minibatch=None)
        xhat, n_actual = sampler()
        self._log_reconstruction(x, xhat, y)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        if y.shape[1] == 1:
            # cast from grayscale to RGB
            y = y.repeat(1, 3, 1, 1)

        kwargs = {"eps": self.t_eps, **kwargs}
        return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn


@ModelRegistry.register("DiscriminativeModel")
class DiscriminativeModel(Model):
    @staticmethod
    def add_argparse_args(parser):
        parser = Model.add_argparse_args(parser)
        parser.add_argument("--discriminative-mode",
            type=str, default="direct", choices=("direct", "residual", "mask"),
            help="Mode to train the model M in. 'direct': xhat=M(y), 'residual': xhat=y-M(y), 'mask': xhat=y*M(y)")
        return parser

    def _init_backbone(self, backbone_name, **init_kwargs):
        backbone_cls = BackboneRegistry.get_by_name(backbone_name)
        # use in_channels=out_channels, as we input only `y` and expect the output to only be `xhat`
        return backbone_cls(
            in_channels=self.num_channels, out_channels=self.num_channels,
            scale_by_sigma=False, conditional=False,
            **init_kwargs
        )

    def __init__(self, discriminative_mode, *args, **kwargs):
        """
        Create a new DiscriminativeModel.

        # Args:
            backbone_name: Name of a registered Backbone DNN class.
            data_module_name: Name of a registered DataModule class.
            optimizer: The name of the optimizer to use (Options: adam, adamw, sgd)
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            loss_type: The type of loss to use. Options are 'mse' (default), 'mae'

            discriminative_mode: The type of discriminative mode to run. 'direct', 'residual' or 'mask'.
        """
        super().__init__(*args, **kwargs)

        # Save discriminative mode
        self.mode = discriminative_mode
        assert self.mode in ('direct', 'residual', 'mask'), f"Unknown discriminative mode: {self.mode}!"

        # Finalize init
        self._finalize_init(**kwargs)

    def forward(self, y):
        backbone_output = self.backbone(y, time=None, sigmas=None)

        if self.mode == 'direct':
            return backbone_output
        elif self.mode == 'residual':
            return y - backbone_output
        elif self.mode == 'mask':
            return y * backbone_output
        else:
            raise NotImplementedError(f"Discriminative mode {self.mode} not implemented")

    def _step(self, batch, batch_idx, mode=None):
        x, y = batch
        xhat = self(y)
        err = xhat - x
        loss = self._get_loss(err, None)
        return loss

    def _log_evaluation(self, batch, batch_idx):
        x, y = batch
        xhat = self(y)
        self._log_reconstruction(x, xhat, y)
