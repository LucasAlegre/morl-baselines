import collections
import random
import yaml
import os
from concurrent.futures import ProcessPoolExecutor

import wandb
import numpy as np
import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.common.utils import reset_wandb_env

WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("sweep_id", "seed", "config")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("hypervolume"))

# Set the number of seeds
num_seeds = 3

# Create an array of seeds
seeds = [random.randint(0, 1000000) for _ in range(num_seeds)]

# Set the count of the sweep agent
count = 5

def train(worker_data):
    # Reset the wandb environment variables
    reset_wandb_env()

    seed = worker_data.seed
    group = worker_data.sweep_id
    config = worker_data.config

    def make_env():
        env = mo_gym.make("minecart-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        return env

    # Create the environments
    env = make_env()
    eval_env = make_env()

    # Create the agent
    agent = Envelope(env, **config, seed=seed, group=group)

    # Launch the agent training
    print("Training started")
    agent.train(
        total_timesteps=1000,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        ref_point=np.array([0, 0, -200.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.98),
        num_eval_weights_for_front=5,
        eval_freq=1000,
        reset_num_timesteps=False,
        reset_learning_starts=False,
        verbose=False
    )
    print("Training finished")

    # Get the hypervolume from the wandb run
    hypervolume = wandb.run.summary["eval/hypervolume"]
    print("Hypervolume: {}".format(hypervolume))

    return WorkerDoneData(hypervolume=hypervolume)

def main():
    # Get the sweep id
    sweep_run = wandb.init()

    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start

    with ProcessPoolExecutor(max_workers=num_seeds) as executor:
        futures = []
        for num in range(num_seeds):
            print("Spinning up worker {}".format(num))
            seed = seeds[num]
            futures.append(executor.submit(train, WorkerInitData(
                sweep_id=sweep_id,
                seed=seed,
                config=dict(sweep_run.config),
            )))

        # Get results from workers
        results = [future.result() for future in futures]

    metrics = [result.hypervolume for result in results]
    print("Metrics: {}".format(metrics))

    # Log the average hypervolume to the sweep run
    sweep_run.log(dict(hypervolume=sum(metrics) / len(metrics)))
    wandb.finish()

# Load the sweep config
config_file = os.path.join(os.path.dirname(__file__), 'sweep_config.yaml')

# Set up the default hyperparameters
with open(config_file) as file:
    sweep_config = yaml.load(file, Loader=yaml.FullLoader)

# Set up the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="MORL-Baselines")

# Run the sweep agent
wandb.agent(sweep_id, function=main, count=count)
