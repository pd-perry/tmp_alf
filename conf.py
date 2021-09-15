import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.trac_algorithm import TracAlgorithm
from alf.algorithms.data_transformer import RewardScaling
from alf.networks import QNetwork

from bsuite import sweep
from alf.environments.suite_bsuite import load

from seed_td import SeedTD

# environment config
alf.config(
    'create_environment', env_name=sweep.CARTPOLE_SWINGUP[1], env_load_fn=load, num_parallel_environments=10)

# reward scaling
alf.config('TrainerConfig', data_transformer_ctor=RewardScaling)
alf.config('RewardScaling', scale=0.01)

# algorithm config
# alf.config('ReluMLP', hidden_layers=(50, 50))
alf.config('QNetwork', fc_layer_params=(50, ))
alf.config(
    'SeedTD',
    optimizer=alf.optimizers.Adam(lr=5e-3, gradient_clipping=10.0))

alf.config(
    'TrainerConfig',
    unroll_length=10,
    algorithm_ctor=SeedTD,
    num_iterations=5000,
    num_checkpoints=5,
    evaluate=True,
    eval_interval=500,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    summary_interval=5,
    epsilon_greedy=0.1)
