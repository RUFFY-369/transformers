# coding=utf-8
# Copyright 2024 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""TD-MPC2 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class TdMpc2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`TdMpc2Model`]. It is used to
    instantiate a TD-MPC2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the standard
    TdMpc2 [ruffy369/tdmpc2-dog-run](https://huggingface.co/ruffy369/tdmpc2-dog-run) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        task (`string`, *optional*, defaults to single task `'dog-run'`):
            The task or task set for which the model training or evaluation has to be performed.
        obs (`string`, defaults to `'state'`):
            The type of observations for the model training/evaluation.
        reward_coef (`float`, defaults to `0.1`):
            The reward prediction coefficient for reward loss.
        value_coef (`float`, defaults to `0.1`):
            The value prediction coefficient for value loss.
        consistency_coef (`int`, defaults to `20`):
            The joint-embedding coefficient for consistency loss.
        rho (`float`, defaults to `0.5`):
            The temporal coefficient for policy loss.
        lr (`float`, defaults to `3e-4`):
            The learning rate for optimizer.
        enc_lr_scale (`float`, defaults to `0.3`):
            The learning rate for encoder's parameters.
        grad_clip_norm (`int`, defaults to `20`):
            The gradient clip norm for model's parameters.
        tau (`float`, defaults to `0.01`):
            The task-agnostic coefficient that balances return maximization and uncertainty minimizations.
        discount_denom (`int`, defaults to `5`):
            The denominator that scales discount linearly with episode length.
        discount_min (`float`, defaults to `0.95`):
            The minimum value of discount factor.
        discount_max (`float`, defaults to `0.995`):
            The maximum value of discount factor.
        buffer_size (`int`, defaults to `1_000_000`):
            The capacity of replay buffer.
        horizon (`int`, defaults to `3`):
            The time horizon for data recording.
        log_std_min (`int`, defaults to `-10`):
            The minimum log std for policy prior.
        log_std_max (`int`, defaults to `2`):
            The maximum log std for policy prior.
        entropy_coef (`float`, defaults to `1e-4`):
            The entropy coefficient for policy prior.
        num_bins (`int`, defaults to `101`):
            The number of bins for discretizing the continuous reward space into a finite number of intervals or categories.
        vmin (`int`, defaults to `-10`):
            The minimum value for reward bins.
        vmax (`int`, defaults to `+10`):
            The maximum value for reward bins.
        model_size (`int`, defaults to `1`):
            The model size for training according to type of task set(single or multi).
        num_enc_layers (`int`, defaults to `2`):
            The number of encoder layers.
        enc_dim (`int`, defaults to `256`):
            The encoder dimension.
        num_channels (`int`, defaults to `32`):
            The number of output channels for encoder.
        mlp_dim (`int`, defaults to `512`):
            The dimensions for MLP module.
        latent_dim (`int`, defaults to `512`):
            The latent dimensions for all components.
        task_dim (`int`, defaults to `0`):
            The task dimensions for multitask training/evaluation.
        num_q (`int`, defaults to `5`):
            The number of Q functions.
        dropout (`float`, defaults to `0.01`):
            The dropout rate for Q functions Ensemble.
        simnorm_dim (`int`, defaults to `8`):
            The dimension for Simplical Normalization.
        multitask (`bool`, defaults to `False`):
            The bool value for multitask training/evaluation.
        tasks (`List[string]`, defaults to `["dog-run"]`):
            The list of task datasets to train on.
        obs_shape (`Dict()`, defaults to `{"state": [223]}`):
            The dictionary for types of observations and their shapes.
        action_dim (`int`, defaults to `38`):
            The dimension of action space.
        episode_length (`int`, defaults to `500`):
            The maximum number of episode steps for the environment.
        obs_shapes (`List[int]`, defaults to `None`):
            The list of observation dimension for multitask training/evaluation.
        action_dims (`List[int]`, defaults to `None`):
            The list of action dimension for multitask training/evaluation.
        episode_lengths (`List[int]`, defaults to `None`):
            The list of maximum number of episode steps for all environments for multitask training/evaluation.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models and 
            isn't used by this model as there are no attention heads).
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 50256):
            Id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            Id of the end of sentence token in the vocabulary.

    Example:

    ```python
    >>> from transformers import TdMpc2Config, TdMpc2Model

    >>> # Initializing a TdMpc2 configuration
    >>> configuration = TdMpc2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = TdMpc2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "td_mpc2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "latent_dim",
        "num_attention_heads": "attention_heads",
        "num_hidden_layers": "num_enc_layers",
        "vocab_size": "vocab_size",
    }

    def __init__(
        self,
        task="dog-run",
        obs="state",
        device="cpu",
        batch_size=256,
        reward_coef=0.1,
        value_coef=0.1,
        consistency_coef=20,
        rho=0.5,
        lr=3e-4,
        enc_lr_scale=0.3,
        grad_clip_norm=20,
        tau=0.01,
        discount_denom=5,
        discount_min=0.95,
        discount_max=0.995,
        buffer_size=1_000_000,
        horizon=3,
        log_std_min=-10,
        log_std_max=2,
        entropy_coef=1e-4,
        num_bins=101,
        vmin=-10,
        vmax=+10,
        model_size= 1,
        num_enc_layers=2,
        enc_dim=256,
        num_channels=32,
        mlp_dim=512,
        latent_dim=512,
        task_dim=0,
        num_q=5,
        dropout=0.01,
        simnorm_dim=8,
        multitask=False,
        tasks=["dog-run"],
        obs_shape={"state": [223]},
        action_dim=38,
        episode_length=500,
        obs_shapes=None,
        action_dims=None,
        episode_lengths=None,
        attention_heads=None,  # no attention heads used in architecture(added for sake of common tests)
        vocab_size=None,  # no transformers used in architecture(added for sake of common tests)
        use_cache=True,
        pad_token_id=1,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        self.task = task
        self.obs = obs
        self.device = device
        self.batch_size = batch_size
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.consistency_coef = consistency_coef
        self.rho = rho
        self.lr = lr
        self.enc_lr_scale = enc_lr_scale
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau
        self.discount_denom = discount_denom
        self.discount_min = discount_min
        self.discount_max = discount_max
        self.buffer_size = buffer_size
        self.horizon = horizon
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.entropy_coef = entropy_coef
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.model_size = model_size
        self.num_enc_layers = num_enc_layers
        self.enc_dim = enc_dim
        self.num_channels = num_channels
        self.mlp_dim = mlp_dim
        self.latent_dim = latent_dim
        self.task_dim = task_dim
        self.num_q = num_q
        self.dropout = dropout
        self.simnorm_dim = simnorm_dim
        self.multitask = multitask
        self.tasks = tasks
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.episode_length = episode_length
        self.obs_shapes = obs_shapes
        self.action_dims = action_dims
        self.episode_lengths = episode_lengths
        self.bin_size = (self.vmax - self.vmin) / (self.num_bins - 1)
        self.attention_heads = attention_heads
        self.vocab_size = vocab_size

        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
