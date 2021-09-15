from functools import partial
from alf.environments.utils import create_environment
import torch
import torch.distributions as td
import unittest

import alf
from alf.utils import common, dist_utils, tensor_utils
from alf.data_structures import StepType, TimeStep
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.networks.q_networks import QNetwork
from alf.networks.relu_mlp import ReluMLP
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from seed_td import SeedTD
from alf.algorithms.rl_algorithm_test import MyEnv


def create_algorithm(env):
    config = TrainerConfig(root_dir="dummy", unroll_length=1)
    obs_spec = alf.TensorSpec((4, ), dtype='float32')
    action_spec = alf.BoundedTensorSpec(
        shape=(), dtype='int32', minimum=0, maximum=1)

    fc_layer_params = (10, 8, 6)

    actor_network = partial(
        ActorDistributionNetwork,
        fc_layer_params=fc_layer_params,
        discrete_projection_net_ctor=alf.networks.CategoricalProjectionNetwork)

    value_network = partial(QNetwork, fc_layer_params=(50, 50))

    qnetwork = partial(ReluMLP, hidden_layers=(50, 50))

    alg = SeedTD(
        observation_spec=obs_spec,
        action_spec=action_spec,
        qnetwork=value_network,
        env=env,
        config=config,
        optimizer=alf.optimizers.Adam(lr=1e-2),
        debug_summaries=True,
        name="SeedTD")
    return alg


class ActorCriticAlgorithmTest(alf.test.TestCase):
    def test_ac_algorithm(self):
        env = create_environment(num_parallel_environments=10)
        # env = MyEnv(1)
        #env = create_environment()
        alg1 = create_algorithm(env)
        time_step = env.step(torch.zeros((10), dtype=torch.int32))
        time_step = common.get_initial_time_step(env)

        iter_num = 1
        # for _ in range(iter_num):
        #     alg1.train_iter()

        time_step = common.get_initial_time_step(env)
        state = alg1.get_initial_predict_state(env.batch_size)
        policy_step = alg1.rollout_step(time_step, state)
        step = alg1.train_step(time_step, state, policy_step.info)
        alg1.calc_loss(step.info)
        print(policy_step[1])
        print("state: ", time_step)
        # logits = policy_step.info.action_distribution.log_prob(
        #     torch.arange(3).reshape(3, 1))

        # print("logits: ", logits[2:])
        # self.assertTrue(torch.all(logits[1, :] > logits[0, :]))
        # self.assertTrue(torch.all(logits[1, :] > logits[2, :]))

        # # global counter is iter_num due to alg1
        self.assertTrue(alf.summary.get_global_counter() == iter_num)

    # def test_ac_algorithm_with_global_counter(self):
    #     env = MyEnv(batch_size=1)
    #     alg2 = create_algorithm(env)
    #     new_iter_num = 3
    #     for _ in range(new_iter_num):
    #         alg2.train_iter()
    #     # new_iter_num of iterations done in alg2
    #     self.assertTrue(alf.summary.get_global_counter() == new_iter_num)


if __name__ == '__main__':
    alf.test.main()
