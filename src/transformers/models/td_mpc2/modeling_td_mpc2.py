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
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
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


def weight_init(m):
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.02, 0.02)
    elif isinstance(m, nn.ParameterList):
        for i,p in enumerate(m):
            if p.dim() == 3: # Linear
                nn.init.trunc_normal_(p, std=0.02) # Weight
                nn.init.constant_(m[i+1], 0) # Bias


def zero_(params):
    """Initialize parameters to zero."""
    for p in params:
        p.data.fill_(0)

# Copied from transformers.models.gpt2.modeling_gpt2.GPT2MLP with GPT2->TdMpc2GPT2
class TdMpc2GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


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

    state_preds: torch.FloatTensor = None
    action_preds: torch.FloatTensor = None
    return_preds: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None


# Copied from transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerPreTrainedModel with DecisionTransformer->TdMpc2,decision_transformer->td_mpc2
class TdMpc2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TdMpc2Config
    base_model_prefix = "td_mpc2"
    main_input_name = "states"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


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

#LAYERSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        self.base_model = copy.deepcopy(modules[0])
        self.base_model.to('meta')
        params, _ = torch.func.stack_module_state(modules)
        self.vmap = torch.vmap(self._call_single_model, (0, 0, None), randomness='different', **kwargs)
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])

    def _call_single_model(self,params, buffers, data):
            return torch.func.functional_call(self.base_model, (params, buffers), (data,))
    
    def forward(self, *args, **kwargs):
        return self.vmap([p for p in self.params], (), *args, **kwargs) 

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

    def __init__(self, in_features,out_features, dropout=0., **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features,**kwargs)
        self.ln = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = self.linear(x)
        if self.dropout:
            x = self.dropout(x)
        return ACT2FN["swish"](self.ln(x))


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)

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
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div_(255.).sub_(0.5)
    
def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)

def enc(config, out={}):
    """
    Returns a dictionary of encoders for each observation in the dict.
    """
    for k in config.obs_shape.keys():
        if k == 'state':
            out[k] = mlp(config.obs_shape[k][0] + config.task_dim, max(config.num_enc_layers-1, 1)*[config.enc_dim], config.latent_dim, act=SimNorm(config))
        elif k == 'rgb':
            out[k] = conv(config.obs_shape[k], config.num_channels, act=SimNorm(config))
        else:
            raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
    return nn.ModuleDict(out)

#####################################################################
class TdMpc2WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.multitask:
            self._task_emb = nn.Embedding(len(config.tasks), config.task_dim, max_norm=1)
            self._action_masks = torch.zeros(len(config.tasks), config.action_dim)
            for i in range(len(config.tasks)):
                self._action_masks[i, :config.action_dims[i]] = 1.
        self._encoder = layers.enc(config)
        self._dynamics = layers.mlp(config.latent_dim + config.action_dim + config.task_dim, 2*[config.mlp_dim], config.latent_dim, act=layers.SimNorm(config))
        self._reward = layers.mlp(config.latent_dim + config.action_dim + config.task_dim, 2*[config.mlp_dim], max(config.num_bins, 1))
        self._pi = layers.mlp(config.latent_dim + config.task_dim, 2*[config.mlp_dim], 2*config.action_dim)
        self._Qs = layers.Ensemble([layers.mlp(config.latent_dim + config.action_dim + config.task_dim, 2*[config.mlp_dim], max(config.num_bins, 1), dropout=config.dropout) for _ in range(config.num_q)])
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(config.log_std_min)
        self.log_std_dif = torch.tensor(config.log_std_max) - self.log_std_min

        
    def to(self, *args, **kwargs):
        """
        Overriding `to` method to also move additional tensors to device.
        """
        super().to(*args, **kwargs)
        if self.config.multitask:
            self._action_masks = self._action_masks.to(*args, **kwargs)
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
        return self

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

    def encode(self, observations, tasks):
        """
        Encodes an observationservation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.config.multitask:
            observations = self.task_emb(observations, tasks)
        if self.config.observations == 'rgb' and observations.ndim == 5:
            return torch.stack([self._encoder[self.config.observations](o) for o in observations])
        return self._encoder[self.config.observations](observations)

    def next(self, z, a, tasks):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.config.multitask:
            z = self.task_emb(z, tasks)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)
    
    def reward(self, z, a, tasks):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.config.multitask:
            z = self.task_emb(z, tasks)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def pi(self, z, tasks):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.config.multitask:
            z = self.task_emb(z, tasks)

        # Gaussian policy prior
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.config.multitask: # Mask out unused action dimensions
            mu = mu * self._action_masks[tasks]
            log_std = log_std * self._action_masks[tasks]
            eps = eps * self._action_masks[tasks]
            action_dims = self._action_masks.sum(-1)[tasks].unsqueeze(-1)
        else: # No masking
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, a, tasks, return_type='min', target=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}

        if self.config.multitask:
            z = self.task_emb(z, tasks)
            
        z = torch.cat([z, a], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)

        if return_type == 'all':
            return out

        Q1, Q2 = out[np.random.choice(self.config.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.config), math.two_hot_inv(Q2, self.config)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2

def action_pred(self, zs, tasks):
        """
        Update policy using a sequence of latent states.
        
        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """

        self.world_model.track_q_grad(False)
        action_pred, _,_,_ = self.world_model.pi(zs, tasks)
    


class TdMpc2Losses(batch,zs,world_model):
    def compute_total_losses(self):
        consistency_loss = 0
        for t in range(self.config.horizon):
            z = self.world_model.next(z, actions[t], tasks)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.config.rho**t
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.world_model.Q(_zs, actions, tasks, return_type='all')
        reward_preds = self.world_model.reward(_zs, actions, tasks)

        #REWARDS PRED HF
        print("reward_preds:",reward_preds)
        
        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.config.horizon):
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.config).mean() * self.config.rho**t
            for q in range(self.config.num_q):
                value_loss += math.soft_ce(qs[q][t], td_targets[t], self.config).mean() * self.config.rho**t
        consistency_loss *= (1/self.config.horizon)
        reward_loss *= (1/self.config.horizon)
        value_loss *= (1/(self.config.horizon * self.config.num_q))
        total_loss = (
            self.config.consistency_coef * consistency_loss +
            self.config.reward_coef * reward_loss +
            self.config.value_coef * value_loss
        )
        return total_loss 

@add_start_docstrings("The TD-MPC2 Model", DECISION_TRANSFORMER_START_DOCSTRING)
# Copied from transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerModel with DecisionTransformer->TdMpc2,edbeeching/decision-transformer-gym-hopper-medium->ruffy369/tdmpc2-dog-run
class TdMpc2Model(TdMpc2PreTrainedModel):
    """

    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.world_model = TdMpc2WorldModel()
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in config.episode_lengths], device='cuda'
        ) if self.config.multitask else self._get_discount(config.episode_length)

        # Initialize weights and apply final processing
        self.post_init()
    
    def _td_target(self, next_z, reward, tasks):
        """
        Compute the TD-target from a reward and the observation at the following time step.
        
        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            tasks (torch.Tensor): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: TD-target.
        """
        pi = self.world_model.pi(next_z, tasks)[1]
        discount = self.discount[tasks].unsqueeze(-1) if self.config.multitask else self.discount
        return reward + discount * self.world_model.Q(next_z, pi, tasks, return_type='min', target=True)

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
            frac = episode_length/self.config.discount_denom
            return min(max((frac-1)/(frac), self.config.discount_min), self.config.discount_max)

    @add_start_docstrings_to_model_forward(DECISION_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TdMpc2Output, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        observations: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        rewards: Optional[torch.FloatTensor] = None,
        tasks: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
            next_z = self.world_model.encode(observations[1:], tasks)
            td_targets = self._td_target(next_z, rewards, tasks)
        
        zs = torch.empty(self.config.horizon+1, self.config.batch_size, self.config.latent_dim, device=self.device)
        z = self.world_model.encode(observations[0], tasks)
        zs[0] = z
        consistency_loss = 0
        for t in range(self.config.horizon):
            z = self.world_model.next(z, actions[t], tasks)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.config.rho**t
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.world_model.Q(_zs, actions, tasks, return_type='all')
        reward_preds = self.world_model.reward(_zs, actions, tasks)

        action_pred = self.action_pred(zs.detach(), tasks)

        # get predictions
        # return_preds = self.predict_return(x[:, 2])  # predict next return given state and actions
        # state_preds = self.predict_state(x[:, 2])  # predict next state given state and actions
        # action_preds = self.predict_action(x[:, 1])  # predict next actions given state
        if not return_dict:
            return tuple(
                v
                for v in [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
                if v is not None
            )

        return TdMpc2Output(
            losses=None,
            action_preds=action_pred,
            reward_preds=reward_preds,
            return_preds=td_targets,
            hidden_states=None,
            attentions=None,
        )
