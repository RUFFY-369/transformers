# coding=utf-8
# Copyright 2024 The HuggingFace Team The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch TdMpc2 model."""

import copy
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_td_mpc2 import TdMpc2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "ruffy369/tdmpc2-dog-run"
_CONFIG_FOR_DOC = "TdMpc2Config"


@dataclass
# Copied from transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerOutput with DecisionTransformer->TdMpc2
class TdMpc2Output(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, state_dim)`):
            Environment state predictions
        action_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, action_dim)`):
            Model action predictions
        return_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Predicted returns for each state
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    action_preds: torch.FloatTensor = None
    losses: torch.FloatTensor = None
    reward_preds: torch.FloatTensor = None
    return_preds: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None


# Copied from transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerPreTrainedModel with DecisionTransformer->TdMpc2,decision_transformer->td_mpc2
class TdMpc2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TdMpc2Config
    base_model_prefix = "td_mpc2"
    main_input_name = "observations"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -0.02, 0.02)
        elif isinstance(module, Ensemble):  # nn.ParameterList
            # Initialize parameters to zero (second last linear layer needs zero initialization on weights)
            for p in module.params[-2]:
                p.data.fill_(0)

            for i, p in enumerate(module.params):
                if p.dim() == 3:  # Linear
                    nn.init.trunc_normal_(p, std=0.02)  # Weight
                    nn.init.constant_(module.params[i + 1], 0)  # Bias
        elif isinstance(module, TdMpc2Model):  # for rewards init
            # Initialize parameters to zero (last linear layer needs zero initialization on weights)
            for p in module.world_model._reward.mlp_layers[-1].weight:
                p.data.fill_(0)


DECISION_TRANSFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~TdMpc2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DECISION_TRANSFORMER_INPUTS_DOCSTRING = r"""
    Args:
        states (`torch.FloatTensor` of shape `(batch_size, episode_length, state_dim)`):
            The states for each step in the trajectory
        actions (`torch.FloatTensor` of shape `(batch_size, episode_length, act_dim)`):
            The actions taken by the "expert" policy for the current state, these are masked for auto regressive
            prediction
        rewards (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):
            The rewards for each state, action
        returns_to_go (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):
            The returns for each state in the trajectory
        timesteps (`torch.LongTensor` of shape `(batch_size, episode_length)`):
            The timestep for each step in the trajectory
        attention_mask (`torch.FloatTensor` of shape `(batch_size, episode_length)`):
            Masking, used to mask the actions when performing autoregressive prediction
"""


# Math operations.


def soft_ce(pred, target, cfg):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim=True)


@torch.jit.script
def log_std(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


@torch.jit.script
def _gaussian_residual(eps, log_std):
    return -0.5 * eps.pow(2) - log_std


@torch.jit.script
def _gaussian_logprob(residual):
    return residual - 0.5 * torch.log(2 * torch.pi)


def gaussian_logprob(eps, log_std, size=None):
    """Compute Gaussian log probability."""
    residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
    if size is None:
        size = eps.size(-1)
    return _gaussian_logprob(residual) * size


@torch.jit.script
def _squash(pi):
    return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi -= _squash(pi).sum(-1, keepdim=True)
    return mu, pi, log_pi


@torch.jit.script
def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx.float()).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.size(0), cfg.num_bins, device=x.device)
    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot


def two_hot_inv(x, cfg):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    DREG_BINS = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
    return symexp(x)


# from functorch import combine_state_for_ensemble


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        self.base_model = copy.deepcopy(modules[0])
        self.base_model.to("meta")
        self.params_dict, _ = torch.func.stack_module_state(modules)
        self.params = nn.ParameterList([nn.Parameter(p) for _, p in self.params_dict.items()])
        self.vmap = torch.func.vmap(
            self._call_single_model, (0, 0, None, None), randomness="different", **kwargs
        )  # randomness='different'

    def _call_single_model(self, params, buffers, data, output_hidden_states: bool = False):
        output, hidden_states = torch.func.functional_call(
            self.base_model, (params, buffers), (data, output_hidden_states)
        )
        return output, hidden_states

    def forward(self, x, output_hidden_states: bool = False, **kwargs):
        params_dict = dict(zip(self.params_dict.keys(), self.params))
        outputs, hidden_states = self.vmap(params_dict, {}, x, output_hidden_states)
        return outputs, hidden_states if output_hidden_states else None


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


class NormedLinear(nn.Module):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, in_features, out_features, dropout=0.0, act=ACT2FN["mish"], **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        self.ln = nn.LayerNorm(out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = self.linear(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div_(255.0).sub_(0.5)


class MLP(nn.Module):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """

    def __init__(self, in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
        super().__init__()
        if isinstance(mlp_dims, int):
            mlp_dims = [mlp_dims]
        dims = [in_dim] + mlp_dims + [out_dim]

        self.mlp_layers = nn.ModuleList()
        for i in range(len(dims) - 2):
            self.mlp_layers.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
        self.mlp_layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
        self.mlp_layers = nn.Sequential(*self.mlp_layers)

    def forward(self, x, output_hidden_states: bool = False):
        hidden_states = () if output_hidden_states else None
        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x)
            hidden_states = hidden_states + (x,) if output_hidden_states else None
        return x, hidden_states if output_hidden_states else torch.empty((0, 0))


class Conv(nn.Module):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    """

    def __init__(self, in_shape, num_channels, act=None):
        super().__init__()
        assert in_shape[-1] == 64  # assumes rgb observations to be 64x64

        self.conv_layers = [
            ShiftAug(),
            PixelPreprocess(),
            nn.Conv2d(in_shape[0], num_channels, 7, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, stride=1),
            nn.Flatten(),
        ]
        if act:
            self.conv_layers.append(act)
        self.conv_layers = nn.Sequential(*self.conv_layers)
        # self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x, output_hidden_states: bool = False):
        # Initialize hidden states here to reset on each forward pass
        hidden_states = () if output_hidden_states else None
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            hidden_states = hidden_states + (x,) if output_hidden_states else None
        return x, hidden_states if output_hidden_states else None


class EncodersDict:
    """
    Returns a dictionary of encoders for each observation in the dict.
    """

    def enc(self, config, out={}):
        self.out = out
        for k in config.obs_shape.keys():
            if k == "state":
                self.out[k] = MLP(
                    config.obs_shape[k][0] + config.task_dim,
                    max(config.num_enc_layers - 1, 1) * [config.enc_dim],
                    config.latent_dim,
                    act=SimNorm(config),
                )
            elif k == "rgb":
                self.out[k] = Conv(config.obs_shape[k], config.num_channels, act=SimNorm(config))
            else:
                raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
        return nn.ModuleDict(self.out)


class RunningScale:
    """Running trimmed scale estimator."""

    def __init__(self, config):
        self.config = config
        self._value = torch.ones(1, dtype=torch.float32, device=torch.device(config.device))
        self._percentiles = torch.tensor([5, 95], dtype=torch.float32, device=torch.device(config.device))

    def state_dict(self):
        return {"value": self._value, "percentiles": self._percentiles}

    def load_state_dict(self, state_dict):
        self._value.data.copy_(state_dict["value"])
        self._percentiles.data.copy_(state_dict["percentiles"])

    @property
    def value(self):
        return self._value.cpu().item()

    def _percentile(self, x):
        x_dtype, x_shape = x.dtype, x.shape
        x = x.view(x.shape[0], -1)
        in_sorted, _ = torch.sort(x, dim=0)
        positions = self._percentiles * (x.shape[0] - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        return (d0 + d1).view(-1, *x_shape[1:]).type(x_dtype)

    def update(self, x):
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
        self._value.data.lerp_(value, self.config.tau)

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        return x * (1 / self.value)

    def __repr__(self):
        return f"RunningScale(S: {self.value})"


class TdMpc2WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        enc_layers = EncodersDict()
        if config.multitask:
            self._task_emb = nn.Embedding(len(config.tasks), config.task_dim, max_norm=1)
            self._action_masks = torch.zeros(len(config.tasks), config.action_dim)
            for i in range(len(config.tasks)):
                self._action_masks[i, : config.action_dims[i]] = 1.0
        self._encoder = enc_layers.enc(config)
        self._dynamics = MLP(
            config.latent_dim + config.action_dim + config.task_dim,
            2 * [config.mlp_dim],
            config.latent_dim,
            act=SimNorm(config),
        )
        self._reward = MLP(
            config.latent_dim + config.action_dim + config.task_dim, 2 * [config.mlp_dim], max(config.num_bins, 1)
        )
        self._pi = MLP(config.latent_dim + config.task_dim, 2 * [config.mlp_dim], 2 * config.action_dim)
        self._Qs = Ensemble(
            [
                MLP(
                    config.latent_dim + config.action_dim + config.task_dim,
                    2 * [config.mlp_dim],
                    max(config.num_bins, 1),
                    dropout=config.dropout,
                ).to(device=config.device)
                for _ in range(config.num_q)
            ]
        )
        self._Qs.base_model.to_empty(device=config.device)
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(config.log_std_min)
        self.log_std_dif = torch.tensor(config.log_std_max) - self.log_std_min

    def track_q_grad(self, mode=True):
        """
        Enables/disables gradient tracking of Q-networks.
        Avoids unnecessary computation during policy optimization.
        This method also enables/disables gradients for task embeddings.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        if self.config.multitask:
            for p in self._task_emb.parameters():
                p.requires_grad_(mode)

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                p_target.data.lerp_(p.data, self.config.tau)

    def task_emb(self, x, tasks):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(tasks, int):
            tasks = torch.tensor([tasks], device=x.device)
        emb = self._task_emb(tasks.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, observations, tasks, output_hidden_states: bool = False):  ############HS##############
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.config.multitask:
            observations = self.task_emb(observations, tasks)
        if self.config.obs == "rgb" and observations.ndim == 5:
            encoded_out = [self._encoder[self.config.obs](o, output_hidden_states) for o in observations]
            if output_hidden_states:
                for out in encoded_out:
                    for hidden_state in out[1]:
                        hidden_state.requires_grad_(True)
            return torch.stack([out[0] for out in encoded_out]), tuple(out[1] for out in encoded_out)
        encoded_out = self._encoder[self.config.obs](observations, output_hidden_states)
        if output_hidden_states:
            for hidden_state in encoded_out[1]:
                hidden_state.requires_grad_(True)
        return encoded_out[0], encoded_out[1]

    def next(self, z, a, tasks, output_hidden_states: bool = False):  #########HS#######
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.config.multitask:
            z = self.task_emb(z, tasks)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z, output_hidden_states)

    def reward(self, z, a, tasks, output_hidden_states: bool = False):  ##############hs###########
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.config.multitask:
            z = self.task_emb(z, tasks)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z, output_hidden_states)

    def pi(self, z, tasks, output_hidden_states: bool = False):  #################HS################3
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.config.multitask:
            z = self.task_emb(z, tasks)

        # Gaussian policy prior
        pi_out, hidden_states_pi = self._pi(z, output_hidden_states)
        if output_hidden_states:
            for hidden_state in hidden_states_pi:
                hidden_state.requires_grad_(True)
        mu, log_strd = pi_out.chunk(2, dim=-1)
        log_strd = log_std(log_strd, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.config.multitask:  # Mask out unused action dimensions
            mu = mu * self._action_masks[tasks]
            log_strd = log_strd * self._action_masks[tasks]
            eps = eps * self._action_masks[tasks]
            action_dims = self._action_masks.sum(-1)[tasks].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_pi = gaussian_logprob(eps, log_strd, size=action_dims)
        pi = mu + eps * log_strd.exp()
        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_strd, hidden_states_pi

    def Q(
        self, z, a, tasks, return_type="min", target=False, output_hidden_states: bool = False
    ):  ################HS###########
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {"min", "avg", "all"}

        if self.config.multitask:
            z = self.task_emb(z, tasks)

        z = torch.cat([z, a], dim=-1)
        out, hidden_states_Qs = (self._target_Qs if target else self._Qs)(z, output_hidden_states)
        if output_hidden_states:
            for hidden_state in hidden_states_Qs:
                hidden_state.requires_grad_(True)

        if return_type == "all":
            return out, hidden_states_Qs

        Q1, Q2 = out[np.random.choice(self.config.num_q, 2, replace=False)]
        Q1, Q2 = two_hot_inv(Q1, self.config), two_hot_inv(Q2, self.config)
        return torch.min(Q1, Q2) if return_type == "min" else (Q1 + Q2) / 2, hidden_states_Qs


class TdMpc2Losses:
    def compute_total_losses(self, config, rewards, z, next_z, reward_preds, td_targets, qs):
        consistency_loss = 0
        for t in range(config.horizon):
            consistency_loss += F.mse_loss(z[t], next_z[t]) * config.rho**t

        reward_loss, value_loss = 0, 0
        for t in range(config.horizon):
            reward_loss += soft_ce(reward_preds[t], rewards[t], config).mean() * config.rho**t
            for q in range(config.num_q):
                value_loss += soft_ce(qs[q][t], td_targets[t], config).mean() * config.rho**t
        consistency_loss *= 1 / config.horizon
        reward_loss *= 1 / config.horizon
        value_loss *= 1 / (config.horizon * config.num_q)
        total_loss = (
            config.consistency_coef * consistency_loss
            + config.reward_coef * reward_loss
            + config.value_coef * value_loss
        )
        return total_loss


@add_start_docstrings("The TD-MPC2 Model", DECISION_TRANSFORMER_START_DOCSTRING)
class TdMpc2Model(TdMpc2PreTrainedModel):
    """

    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.world_model = TdMpc2WorldModel(config)
        self.scale = RunningScale(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length / self.config.discount_denom
        return min(max((frac - 1) / (frac), self.config.discount_min), self.config.discount_max)

    def _td_target(self, next_z, reward, tasks, output_hidden_states: bool = False):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            tasks (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: TD-target.
        """
        pi = self.world_model.pi(next_z, tasks, output_hidden_states)[1]

        discount = (
            torch.tensor(
                [self._get_discount(ep_len) for ep_len in self.config.episode_lengths], device=self.config.device
            )
            if self.config.multitask
            else self._get_discount(self.config.episode_length)
        )

        discount = discount[tasks].unsqueeze(-1) if self.config.multitask else discount
        Q_out = self.world_model.Q(
            next_z, pi, tasks, return_type="min", target=True, output_hidden_states=output_hidden_states
        )

        return reward + discount * Q_out[0], Q_out[1]

    def update_pi(self, zs, task, output_hidden_states: bool = False):
        """
        Update policy using a sequence of latent states.

        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        self.world_model.track_q_grad(False)
        action_pred, pis, log_pis, _, hidden_states_pi = self.world_model.pi(zs, task, output_hidden_states)
        if output_hidden_states:
            for hidden_state in hidden_states_pi:
                hidden_state.requires_grad_(True)
        qs, hidden_states_upd_pi_Q = self.world_model.Q(
            zs, pis, task, return_type="avg", output_hidden_states=output_hidden_states
        )
        if output_hidden_states:
            for hidden_state in hidden_states_upd_pi_Q:
                hidden_state.requires_grad_(True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.config.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.config.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
        torch.nn.utils.clip_grad_norm_(self.world_model._pi.parameters(), self.config.grad_clip_norm)
        self.world_model.track_q_grad(True)

        return pi_loss, action_pred, hidden_states_pi, hidden_states_upd_pi_Q

    @add_start_docstrings_to_model_forward(DECISION_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TdMpc2Output, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        observations: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        rewards: Optional[torch.FloatTensor] = None,
        tasks: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.FloatTensor], TdMpc2Output]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import TdMpc2Model
        >>> import torch

        >>> model = TdMpc2Model.from_pretrained("ruffy369/tdmpc2-dog-run")
        >>> # evaluation
        >>> model = model.to(device)
        >>> model.eval()

        >>> env = gym.make("Hopper-v3")
        >>> state_dim = env.observation_space.shape[0]
        >>> act_dim = env.action_space.shape[0]

        >>> state = env.reset()
        >>> states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
        >>> actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
        >>> rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
        >>> target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
        >>> timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        >>> attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

        >>> # forward pass
        >>> with torch.no_grad():
        ...     state_preds, action_preds, return_preds = model(
        ...         states=states,
        ...         actions=actions,
        ...         rewards=rewards,
        ...         returns_to_go=target_return,
        ...         timesteps=timesteps,
        ...         attention_mask=attention_mask,
        ...         return_dict=False,
        ...     )
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # SHAPE BUFFER: torch.Size([4, 256, 223])(observations) torch.Size([3, 256, 38])(action) torch.Size([3, 256, 1])(reward) None(task):
        # # timesteps-1,batch_size,observation_dim_dog_run, timesteps-1,batch_size,action_dim_dog_run, timesteps-1,batch_size

        with torch.no_grad():
            next_z, _ = self.world_model.encode(observations[1:], tasks, output_hidden_states)
            td_targets, _ = self._td_target(next_z, rewards, tasks, output_hidden_states)

        losses_td_mpc2 = TdMpc2Losses()
        zs = torch.empty(self.config.horizon + 1, self.config.batch_size, self.config.latent_dim, device=self.device)
        z, hidden_states_z = self.world_model.encode(observations[0], tasks, output_hidden_states)
        if output_hidden_states:
            for hidden_state in hidden_states_z:
                hidden_state.requires_grad_(True)
        zs[0] = z

        for t in range(self.config.horizon):
            z, hidden_states_next_state = self.world_model.next(z, actions[t], tasks, output_hidden_states)
            if output_hidden_states:
                for hidden_state in hidden_states_next_state:
                    hidden_state.requires_grad_(True)
            zs[t + 1] = z

        z_copy = zs.clone()
        # Predictions
        _zs = zs[:-1]
        qs, hidden_states_actions_Q = self.world_model.Q(
            _zs, actions, tasks, return_type="all", output_hidden_states=output_hidden_states
        )
        reward_preds, hidden_states_rewards = self.world_model.reward(_zs, actions, tasks, output_hidden_states)
        if output_hidden_states:
            for hidden_state in hidden_states_rewards:
                hidden_state.requires_grad_(True)

        self.world_model.track_q_grad(False)

        total_loss = losses_td_mpc2.compute_total_losses(
            self.config, rewards, z_copy, next_z, reward_preds, td_targets, qs
        )
        pi_loss, action_pred, hidden_states_pi, hidden_states_upd_pi_Q = self.update_pi(
            zs.detach(), tasks, output_hidden_states
        )
        total_model_loss = (total_loss, pi_loss)

        all_hidden_states = (
            hidden_states_z
            + hidden_states_next_state
            + hidden_states_actions_Q
            + hidden_states_rewards
            + hidden_states_pi
            + hidden_states_upd_pi_Q
            if output_hidden_states
            else None
        )

        if not return_dict:
            return tuple(
                v
                for v in [
                    action_pred,
                    total_model_loss,
                    reward_preds,
                    td_targets,
                    all_hidden_states,
                    None,
                ]
                if v is not None
            )

        return TdMpc2Output(
            action_preds=action_pred,
            losses=total_model_loss,
            reward_preds=reward_preds,
            return_preds=td_targets,
            hidden_states=all_hidden_states,
            attentions=None,
        )
