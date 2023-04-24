import multiprocessing
import collections
import random
import yaml
import os
import time

import wandb
import numpy as np
import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.envelope.envelope import Envelope

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("sweep_id", "seed", "config")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("hypervolume"))

# TODO:
# Define an array of seeds on top level and reuse them for each sweep iteration
# Use seed_everything() to set the seed for each worker
# Move a function to reset the wandb environment variables to common

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

def train(sweep_q, worker_q):
    # Reset the wandb environment variables
    reset_wandb_env()

    # Get the worker data
    worker_data = worker_q.get()
    config = worker_data.config
    seed = worker_data.seed
    group = worker_data.sweep_id

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
        total_timesteps=100000,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        ref_point=np.array([0, 0, -200.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.98),
        num_eval_weights_for_front=100,
        eval_freq=50000,
        reset_num_timesteps=False,
        reset_learning_starts=False,
    )
    print("Training finished")

    # Get the hypervolume from the wandb run
    hypervolume = wandb.run.summary["eval/hypervolume"]
    print("Hypervolume: {}".format(hypervolume))

    # Send the hypervolume back to the main process
    sweep_q.put(WorkerDoneData(hypervolume=hypervolume))

def main():
    # Set the number of seeds
    num_seeds = 3

    # Create an array of seeds
    seeds = [random.randint(0, 1000000) for _ in range(num_seeds)]

    # Get the sweep id
    sweep_run = wandb.init()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers = []
    for num in range(num_seeds):
        print("Spinning up worker {}".format(num))
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=train, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    metrics = []
    for num in range(num_seeds):
        print("Starting worker {}".format(num))
        seed = seeds[num]
        worker = workers[num]
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                seed=seed,
                config=dict(sweep_run.config),
            )
        )
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        metrics.append(result.hypervolume)

    print("Metrics: {}".format(metrics))

    # Log the average hypervolume to the sweep run
    sweep_run.log(dict(hypervolume=sum(metrics) / len(metrics)))
    wandb.finish()

# Set up the default hyperparameters
with open('./sweep_config.yaml') as file:
    sweep_config = yaml.load(file, Loader=yaml.FullLoader)

# Set up the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="MORL-Baselines")

# Run the sweep agent
wandb.agent(sweep_id, function=main, count=5)

