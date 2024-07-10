<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TD-MPC2

## Overview

The TD-MPC2 model was proposed in [TD-MPC2:Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828) by Nicklas Hansen, Hao Su, Xiaolong Wang.

The abstract from the paper is the following:

*TD-MPC is a model-based reinforcement learning (RL) algorithm that performs local trajectory optimization in the latent space of a learned implicit (decoderfree) world model. In this work, we present TD-MPC2: a series of improvements upon the TD-MPC algorithm. We demonstrate that TD-MPC2 improves significantly over baselines across 104 online RL tasks spanning 4 diverse task domains, achieving consistently strong results with a single set of hyperparameters. We further show that agent capabilities increase with model and data size, and successfully train a single 317M parameter agent to perform 80 tasks across multiple task domains, embodiments, and action spaces. We conclude with an account of lessons, opportunities, and risks associated with large TD-MPC2 agents.*

Tips:

The hugging face version of the model provides the architecture of the original model for training and inference with the exact same output of precision more than 1e-3. The batch provided for training and inference is same as given in the original model, i.e., [observations,actions,rewards,tasks/task embeddings]. It depends on you if you want to use the same method of data collection from env as provided in the original code or use or your own as the this version supports both. The model is multitask model for performing various tasks of different domains,action spaces or can be used as single task model for just one env. If you want to use it, provide the batch consisting of [observations(state space provided from env),actions,rewards,tasks/task embeddings](from the env).

This model was contributed by [ruffy369](https://huggingface.co/ruffy369). The original code can be found [here](https://github.com/nicklashansen tdmpc2).


## TdMpc2Config

[[autodoc]] TdMpc2Config

## TdMpc2Model

[[autodoc]] TdMpc2Model
    - forward
