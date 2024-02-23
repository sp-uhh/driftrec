import abc
import random

import torch
import numpy as np

from driftrec.util.registry import Registry


PredictorRegistry = Registry("Predictor")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    def debug_update_fn(self, x, t, *args):
        raise NotImplementedError(f"Debug update function not implemented for predictor {self}.")


@PredictorRegistry.register('euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    """Standard Euler-Maruyama scheme. 1 NFE per step."""
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        f, g = self.rsde.sde(x, t, *args)
        x_mean = x + f * dt
        x = x_mean + g[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@PredictorRegistry.register('euler_heun')
class EulerHeunPredictor(Predictor):
    """Euler-Heun scheme for Ito SDEs, as presented in arXiv:1210.0933. 2 NFE per step."""
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args):
        h = 1. / self.rsde.N
        assert (t == t[0]).all(), "only works for constant t for all examples for now"

        f1, g1 = self.rsde.sde(x, t, *args)
        sqrth = np.sqrt(h)
        dwk = sqrth * torch.randn_like(x)
        if t[0]-h > 0.0:
            # `sk` is a funny little parameter, see arXiv:1210.0933 for an explanation
            sk = random.choice((-1, 1))
            k1_mean = -h*f1
            k1 = k1_mean + (dwk + sk*sqrth)*g1[:, None, None, None]

            f2, g2 = self.rsde.sde(x+k1, t-h, *args)
            k2_mean = -h*f2
            k2 = k2_mean + (dwk - sk*sqrth)*g2[:, None, None, None]
            x1 = x + 0.5*(k1 + k2)
            x1_mean = x + 0.5*(k1_mean + k2_mean)
        else:
            # Run Euler-Maruyama in the last step, since t+h would become negative
            x1_mean = x - h*f1
            x1 = x1_mean + g1[:, None, None, None] * dwk

        return x1, x1_mean


@PredictorRegistry.register('reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args):
        f, g = self.rsde.discretize(x, t, *args)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + g[:, None, None, None] * z
        return x, x_mean


@PredictorRegistry.register('none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def update_fn(self, x, t, *args):
        return x, x
