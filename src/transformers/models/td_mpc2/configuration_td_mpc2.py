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
    TdMpc2 architecture. Many of the config options are used to instatiate the GPT2 model that is used as
    part of the architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        state_dim (`int`, *optional*, defaults to 17):
            The state size for the RL environment
        act_dim (`int`, *optional*, defaults to 4):
            The size of the output action space
        hidden_size (`int`, *optional*, defaults to 128):
            The size of the hidden layers
        max_ep_len (`int`, *optional*, defaults to 4096):
            The maximum length of an episode in the environment
        action_tanh (`bool`, *optional*, defaults to True):
            Whether to use a tanh activation on action prediction
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`TdMpc2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_layer (`int`, *optional*, defaults to 3):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. If unset, will default to 4 times `n_embd`.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

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
        num_bins=101,  # discretizing the continuous reward space into a finite number of intervals or categories.
        vmin=-10,
        vmax=+10,
        # model_size= 1,
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
        attention_heads=None,  # no attention heads used in architecture
        vocab_size=None,  # no transformers used in architecture
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
