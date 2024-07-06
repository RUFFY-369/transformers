# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch TdMpc2 model."""

import inspect
import unittest

from transformers import TdMpc2Config, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from transformers import TdMpc2Model


class TdMpc2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=8,
        horizon=3,
        act_dim=38,
        state_dim=223,
        tasks=None, #For single task its None unless multitasking training is done
        # hidden_size=23,
        # max_length=11,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.horizon = horizon
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.tasks = tasks
        # self.hidden_size = hidden_size
        # self.max_length = max_length
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # Generate a random float between the range from -1216.0 to 2040.0
        states =  (floats_tensor((self.horizon+1,self.batch_size, self.state_dim)) * 3256.0) - 1216.0 #(states*(2040.0 + 1216.0))-1216.0
        #Generate a random float between the range from -1.0 to 1.0
        actions = (floats_tensor((self.horizon,self.batch_size, self.act_dim)) * 2) - 1
        rewards = floats_tensor((self.horizon,self.batch_size, 1))
        tasks = self.tasks

        return (
            config,
            states,
            actions,
            rewards,
            tasks,
        )

    def get_config(self):
        return TdMpc2Config(
            batch_size=self.batch_size,
            horizon=self.horizon,
        )

    def create_and_check_model(
        self,
        config,
        states,
        actions,
        rewards,
        tasks,
    ):
        model = TdMpc2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(states, actions, rewards, tasks)

        action_pred_expected_shape = torch.Size((actions.shape[0]+1,actions.shape[1],actions.shape[2])) # first index:horizon+1
        reward_pred_expected_shape = torch.Size((rewards.shape[0],rewards.shape[1],config.num_bins))
        
        self.parent.assertEqual(result.action_preds.shape, action_pred_expected_shape)
        self.parent.assertEqual(result.reward_preds.shape, reward_pred_expected_shape)
        #td targets
        self.parent.assertEqual(result.return_preds.shape, rewards.shape)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            states,
            actions,
            rewards,
            tasks,
        ) = config_and_inputs
        inputs_dict = {
            "observations": states,
            "actions": actions,
            "rewards": rewards,
            "tasks": tasks,
        }
        return config, inputs_dict


@require_torch
class TdMpc2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TdMpc2Model,) if is_torch_available() else ()
    all_generative_model_classes = ()
    # Ignoring of a failing test from GenerationTesterMixin, as the model does not use inputs_ids
    test_generate_without_input_ids = False

    # Ignoring of a failing tests from ModelTesterMixin, as the model does not implement these features
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_attention_outputs = False
    test_hidden_states_output = False
    test_inputs_embeds = False
    test_model_common_attributes = False
    test_gradient_checkpointing = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = TdMpc2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TdMpc2Config, hidden_size=512)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "ruffy369/tdmpc2-dog-run"
        model = TdMpc2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "observations",
                "actions",
                "rewards",
                "tasks",
            ]

            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)


@require_torch
class TdMpc2ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_autoregressive_prediction(self):
        """
        An integration test that performs predicts outcomes (returns) conditioned on a sequence of actions, joint-embedding prediction
        (for multitask dataset,this test use single task), rewards, and TD-learning without decoding observations from a sequence of 
        observations,actions,rewards and task embeddings. Test is performed over two timesteps.


        """

        NUM_STEPS = 1  # number of steps of prediction we will perform
        batch_size = 1
        model = TdMpc2Model.from_pretrained("ruffy369/tdmpc2-dog-run")
        model = model.to(torch_device)
        model.eval()
        config = model.config
        torch.manual_seed(0)

        expected_outputs = torch.tensor(
            [[0.242793, -0.28693074, 0.8742613], [0.67815274, -0.08101085, -0.12952147]], device=torch_device
        )
        

        states =  (torch.rand(config.horizon+1,batch_size, config.obs_shape['state']) * 3256.0) - 1216.0 #(states*(2040.0 + 1216.0))-1216.0
        actions = (torch.rand(config.horizon,batch_size,config.action_dim) * 2) - 1
        rewards = torch.rand(config.horizon,batch_size, 1)
        task = None
        
        for step in range(NUM_STEPS):
            with torch.no_grad():
                model_pred = model(
                    observations=states,
                    actions=actions,
                    rewards=rewards,
                    tasks = task,
                    return_dict=True,
                )

            actions_expected_shape = torch.Size((actions.shape[0]+1, actions.shape[1],actions.shape[2]))
            self.assertEqual(model_pred.action_preds.shape, actions_expected_shape)
            self.assertTrue(torch.allclose(model_pred.action_preds, expected_outputs[step], atol=1e-4))