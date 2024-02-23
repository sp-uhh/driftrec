"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc

import numpy as np
from numpy import pi
from scipy.special import dawsn
from scipy.integrate import quad
from scipy.interpolate import interp1d
import torch

from driftrec.util.registry import Registry


SDERegistry = Registry("SDE")


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    def prior_sampling(self, y, T=None, prior_mean=None):
        T = T if T is not None else self.T
        std = self._std(torch.ones((y.shape[0],), device=y.device) * T)
        z = torch.randn_like(y)
        if prior_mean is not None:
            if prior_mean == 'zero':
                z = z - z.mean(dim=(2, 3), keepdim=True)
            else:
                raise ValueError(f"Invalid prior mean: {prior_mean}")
        x_T = y + z*std[:, None, None, None]
        return x_T

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, t, *args):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t, *args)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                score = score_model(x, t, *args)
                score_drift = -sde_diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                return (sde_drift + score_drift), diffusion

            def discretize(self, x, t, *args):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, *args)
                rev_f = f - G[:, None, None, None] ** 2 * score_model(x, t, *args) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass


@SDERegistry.register("ouve")
class OUVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=30, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--gamma", type=float, default=2.0, help="The constant stiffness of the Ornstein-Uhlenbeck process. 2.0 by default.")
        parser.add_argument("--sigma-min", type=float, default=0.01, help="The minimum sigma to use. 0.01 by default.")
        parser.add_argument("--sigma-max", type=float, default=0.3, help="The maximum sigma to use. 0.3 by default.")
        parser.add_argument("--linear-schedule", action="store_true", help="Pass to use a linear noise schedule rather than an exponential one.")
        return parser

    def __str__(self):
        return (
            f"OUVESDE(gamma={self.gamma:.2f}, sigma_min={self.sigma_min:.2g}, sigma_max={self.sigma_max:.2g}, "
            f"linear_schedule={self.linear_schedule}, sde_n={self.N})"
        )

    def __init__(self, gamma, sigma_min, sigma_max, linear_schedule=False, sde_n=1000, **unused_kwargs):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = gamma (y-x) dt + g(t) dw

        with the standard exponential schedule

        g(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 (gamma + log(sigma_max/sigma_min)))

        or, when passing `linear_schedule=True`,

        g(t) = TODO

        Args:
            gamma: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(sde_n)
        self.gamma = gamma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = sde_n
        self.linear_schedule = linear_schedule
        if self.linear_schedule:
            s0, s1, g = self.sigma_min, self.sigma_max, self.gamma
            self._delta_sigma     = s1-s0
            self._delta_sigma_min = s1-s0 - 2*g*s0
            self._delta_sigma_max = s1-s0 - 2*g*s1
            self._num = 4*np.exp(2*g) * g**2 * s1
            self._denom = self._delta_sigma_min - np.exp(2*g)*self._delta_sigma_max

    def copy(self):
        return OUVESDE(self.gamma, self.sigma_min, self.sigma_max, linear_schedule=self.linear_schedule, sde_n=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t, y, *ignored_args):
        drift = self.gamma * (y - x)
        if self.linear_schedule:
            s0, s1 = self.sigma_min, self.sigma_max
            diffusion = torch.sqrt((s1 + t*(s1-s0)) * self._num/self._denom)
        else:
            _sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            diffusion = _sigma * np.sqrt(2 * (self.gamma + np.log(self.sigma_max / self.sigma_min)))
        return drift, diffusion

    def _interp(self, t):
        return torch.exp(-self.gamma * t)[:, None, None, None]

    def _mean(self, x0, t, y):
        interp = self._interp(t)
        return interp * x0 + (1 - interp) * y

    def _std(self, t):
        # This is a solution P(t) to the variance ODE in our derivations, after choosing drift and diffusion
        # as in self.sde() and setting the initial value P(0) = 0.
        s0, s1, g = self.sigma_min, self.sigma_max, self.gamma
        if self.linear_schedule:
            ds, dsm = self._delta_sigma, self._delta_sigma_min
            prefac = np.exp(2*g) * s1 / self._denom
            return torch.sqrt(prefac * (dsm*torch.exp(-2*g*t) + 2*g*(t*ds + s0) - ds))
        else:
            return torch.sqrt(s0**2 * ((s1/s0)**(2*t) - torch.exp(-2*g*t)))

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)


@SDERegistry.register("t2ve")
class T2VESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=30, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--gamma", type=float, default=2.0, help="The constant stiffness of the Ornstein-Uhlenbeck process. 2.0 by default.")
        parser.add_argument("--sigma-min", type=float, default=0.01, help="The minimum sigma to use. 0.01 by default.")
        parser.add_argument("--sigma-max", type=float, default=0.3, help="The maximum sigma to use. 0.3 by default.")
        parser.add_argument("--linear-schedule", action="store_true", help="Pass to use a linear noise schedule rather than an exponential one.")
        return parser

    def __str__(self):
        return (
            f"T2VE(gamma={self.gamma:.2f}, sigma_min={self.sigma_min:.2g}, sigma_max={self.sigma_max:.2g}, "
            f"sde_n={self.N})"
        )

    def __init__(self, gamma, sigma_min, sigma_max, sde_n=1000, **unused_kwargs):
        """Construct an t^2 Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = [gamma * t * (y-x)] dt + g(t) dw

        with the standard exponential schedule

        g(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(normfac)

        where `normfac` is a messed up expression determined by Mathematica to ensure the solved variance sigma(t)^2
        is equal to sigma_max at t=1 :)

        Args:
            gamma: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(sde_n)
        self.gamma = gamma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = sde_n

        s0, s1, g = self.sigma_min, self.sigma_max, self.gamma
        self._logsig = np.log(s1) - np.log(s0)
        self._daws1 = dawsn(self._logsig / np.sqrt(g))
        self._daws2 = dawsn((g + self._logsig) / np.sqrt(g))
        self._denom = (s0**2 * self._daws1) - (s1**2 * np.exp(g) * self._daws2)
        self._diff_constant = np.sqrt(-np.exp(g) * s1**2 * np.sqrt(g) / self._denom)

    def copy(self):
        return T2VESDE(self.gamma, self.sigma_min, self.sigma_max, sde_n=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t, y, *ignored_args):
        s0, s1, g = self.sigma_min, self.sigma_max, self.gamma
        drift = g * t[:, None, None, None] * (y - x)
        diffusion = (s0 * (s1 / s0)**t) * self._diff_constant
        return drift, diffusion

    def _interp(self, t):
        return torch.exp(-self.gamma * t**2 / 2)[:, None, None, None]

    def _mean(self, x0, t, y):
        interp = self._interp(t)
        return interp * x0 + (1 - interp) * y

    def _std(self, t):
        # This is a solution P(t) to the variance ODE in our derivations, after choosing drift and diffusion
        # as in self.sde() and setting the initial value P(0) = 0.
        s0, s1, g = self.sigma_min, self.sigma_max, self.gamma
        daws1 = self._daws1
        logsig = self._logsig
        # the time-dependent Dawson function term:
        daws3 = torch.from_numpy(dawsn((t.cpu().numpy()*g + logsig) / np.sqrt(g))).to(t.device)
        return torch.sqrt(
            torch.exp(g - g*t**2) * s1**2 * s0**2 * (
                daws1 - torch.exp(t * (t*g + 2*self._logsig)) * daws3
            )
            /
            self._denom
        )

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)


@SDERegistry.register("cosve")
class CosVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=30, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--gamma", type=float, default=1.0, help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.0 by default.")
        parser.add_argument("--sigma-min", type=float, default=0.01, help="The minimum sigma to use. 0.01 by default.")
        parser.add_argument("--sigma-max", type=float, default=1.0, help="The maximum sigma to use. 1.0 by default.")
        return parser

    def __str__(self):
        return (
            f"CosVESDE(gamma={self.gamma:.2f}, sigma_min={self.sigma_min:.2g}, sigma_max={self.sigma_max:.2g}, "
            f"sde_n={self.N})"
        )

    def __init__(self, gamma, sigma_min, sigma_max, linear_schedule=False, sde_n=1000, **unused_kwargs):
        """Construct a cosine-stiffness Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = gamma (1-cos(pi t)) (y-x) dt + g(t) dw

        with the standard exponential schedule

        g(t) = sigma_min (sigma_max/sigma_min)^t * normfac

        where normfac is determined numerically via scipy.integrate.quad to ensure that the variance of the process at
        t=1 is equal to sigma_max.

        Note that the variance of the process is also solved for numerically via scipy.integrate.quad and values are
        provided via cubic spline interpolation, so these are necessarily approximations. These values are all
        calculated at construction time, so the (small) computational cost is incurred only once.

        Args:
            gamma: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(sde_n)
        self.gamma = gamma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = sde_n

        integrand = lambda k1: np.exp(2*gamma*(k1 - np.sin(pi * k1)/pi)) * (sigma_max/sigma_min)**(2*k1) * sigma_min**2
        self.normfac = self.sigma_max / np.sqrt(np.exp(-2*gamma) * quad(integrand, 0., 1.)[0])
        time_cache = np.linspace(0, 1, 1000)
        var_cache = np.array([
            self.normfac**2 * np.exp(-2*t*gamma + 2*gamma*np.sin(pi*t)/pi) * quad(integrand, 0., t)[0]
            for t in time_cache
        ])
        self._time_cache = time_cache
        self._std_cache = np.sqrt(var_cache)
        self._std_func = interp1d(self._time_cache, self._std_cache, kind="cubic")

    def copy(self):
        return CosVESDE(self.gamma, self.sigma_min, self.sigma_max, sde_n=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t, y, *ignored_args):
        drift = self.gamma * (1 - torch.cos(t[:,None,None,None] * pi)) * (y - x)
        diffusion = self.normfac * self.sigma_min * (self.sigma_max / self.sigma_min)**t
        return drift, diffusion

    def _interp(self, t):
        return torch.exp(-self.gamma * (t - torch.sin(pi*t)/pi))[:, None, None, None]

    def _mean(self, x0, t, y):
        interp = self._interp(t)
        return interp * x0 + (1 - interp) * y

    def _std(self, t):
        return torch.from_numpy(self._std_func(t.cpu().numpy()).astype(np.float32)).to(t.device)

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)
