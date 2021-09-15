from alf.algorithms.config import TrainerConfig
from alf.networks.q_networks import QNetwork
from alf.networks.value_networks import ValueNetwork
from alf.tensor_specs import TensorSpec
import torch
import torch.nn as nn
import numpy as np
import random

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.networks.relu_mlp import ReluMLP
from alf.data_structures import AlgStep, LossInfo, namedtuple, TimeStep
from alf.utils import value_ops, tensor_utils
from alf.utils.losses import element_wise_squared_loss

TInfo = namedtuple("TInfo", ["state", "action", "reward", "value",
                             "prev_obs", "step_type", "discount"], default_value=())

TState = namedtuple("TState", ["prev_obs"], default_value=())


@alf.configurable
class SeedTD(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec=TensorSpec(()),
                 action_spec=TensorSpec(()),
                 reward_spec=TensorSpec(()),
                 qnetwork=QNetwork,
                 learning_rate=0.001,
                 v=0.005,
                 regularization_factor=10,
                 gamma=0.98,
                 e=0.1,
                 env=None,
                 optimizer=None,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="SeedTD"):
        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         train_state_spec=TState(prev_obs=observation_spec),
                         reward_spec=reward_spec,
                         env=env,
                         optimizer=optimizer,
                         config=config,
                         debug_summaries=debug_summaries,
                         name=name)

        self.num_actions = action_spec.maximum - action_spec.minimum + 1
        qnetwork = QNetwork(input_tensor_spec=observation_spec, action_spec=action_spec)
        self._network = qnetwork.make_parallel(10)
        self._initial_params = [self.generate_q() for _ in range(10)]
        self._lr = learning_rate
        self._v = v
        self._regularization_factor = regularization_factor
        self._gamma = gamma
        self._epsilon_greedy = e
        self._config = config

    def rollout_step(self, input: TimeStep, state: TState):
        value, _ = self._network(input.observation)

        e = random.randint(0, 100)

        if e > self._epsilon_greedy*100:
            action = torch.argmax(value, dim=2)
            action = torch.diagonal(action, 0)
        else:
            action = torch.tensor([random.randint(
                self._action_spec.minimum, self._action_spec.maximum) for i in range(input.observation.shape[0])])
        

        return AlgStep(output=action,
                       state=state,
                       info=TInfo(state=input.observation,
                                  action=input.prev_action,
                                  reward=input.reward,
                                  step_type=input.step_type,
                                  discount=input.discount))

    def train_step(self, input: TimeStep, state, rollout_info: TInfo):
        value, _ = self._network(state.prev_obs)
        action = torch.argmax(value, dim=2)
        action = torch.diagonal(action, 0)
        new_value = torch.diagonal(value, 0)
        prev_action = input.prev_action.to(torch.int64)
        new_value = new_value.t().gather(1, prev_action.view(-1,1))
        new_value = torch.squeeze(new_value.t())

        return AlgStep(output=action,
                       state=TState(prev_obs=input.observation),
                       info=TInfo(discount=input.discount,
                                  step_type=input.step_type,
                                  reward=input.reward,
                                  value=new_value))

    
    def generate_q(self):
        glorot_layers = []
        first_layer = torch.zeros((50, 8))
        first_layer_glorot = torch.nn.init.xavier_normal(first_layer)

        third_layer = torch.zeros((self.num_actions, 50))
        third_layer_glorot = torch.nn.init.xavier_normal(third_layer)

        glorot_layers.append(first_layer_glorot)
        glorot_layers.append(third_layer_glorot)

        return glorot_layers
    
    def calc_regularization(self, generated, trained):
        regularization = torch.zeros((10))

        #TODO: get trained layers weights
        # for i in range(len(trained)):
        #     print(i)
        #     regularization += torch.norm(
        #         generated[i] - trained[i].weight) ** 2

        for i in range(10):
            reg = 0
            for j in range(len(generated[0])):
                reg += torch.norm(generated[i][j]) ** 2
            regularization[i] = reg
        return regularization

    def calc_loss(self, info):

        gaussian_noise = torch.normal(0, self._v, info.reward.shape)

        # trained = self._network._encoding_net._fc_layers
        trained = self._network._encoding_net._pnet
        generated = self._initial_params

        returns = value_ops.one_step_discounted_return(
            info.reward, info.value, info.step_type, info.discount * self._gamma)
        returns = tensor_utils.tensor_extend(returns, info.value[-1])
        returns = returns + gaussian_noise

        loss = element_wise_squared_loss(returns, info.value)
        regularization = self.calc_regularization(generated, trained)

        loss = loss + 1/self._regularization_factor * regularization

        return LossInfo(loss=loss)
