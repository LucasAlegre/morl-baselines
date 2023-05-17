import argparse
import random
import yaml
import os
from dataclasses import dataclass

from concurrent.futures import ProcessPoolExecutor

import wandb
import numpy as np
import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.common.utils import reset_wandb_env

from morl_baselines.common.experiments import ALGOS, ENVS_WITH_KNOWN_PARETO_FRONT, StoreDict

@dataclass
class WorkerInitData:
    sweep_id: str
    seed: int
    config: dict
    worker_num: int

@dataclass
class WorkerDoneData:
    hypervolume: float

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="Name of the algorithm to run", choices=ALGOS.keys(), required=True)
    parser.add_argument("--env-id", type=str, help="MO-Gymnasium id of the environment to run", required=True)
    parser.add_argument(
        "--ref-point", type=float, nargs="+", help="Reference point to use for the hypervolume calculation", required=True
    )

    # parser.add_argument("--seed", type=int, help="Random seed to use", default=42)
    # parser.add_argument("--sweep-id", type=str, help="Sweep id to use for the sweep", required=False)

    parser.add_argument("--wandb-entity", type=str, help="Wandb entity to use for the sweep", required=False)
    parser.add_argument("--project-name", type=str, help="Project name to use for the sweep", default="MORL-Baselines")

    parser.add_argument("--sweep-count", type=int, help="Number of times to run the sweep", default=10)
    parser.add_argument("--num-seeds", type=int, help="Number of seeds to use for the sweep", default=3)

    parser.add_argument(
        "--train-hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Override hyperparameters to use for the train method algorithm. Example: --train-hyperparams num_eval_weights_for_front:10 timesteps_per_iter:10000",
        default={},
    )

    args = parser.parse_args()

    parser.add_argument("--config-name", type=str, help="Name of the config to use for the sweep.", default=f"{args.algo}.yaml")

    args = parser.parse_args()

    return args

def train(worker_data: WorkerInitData) -> WorkerDoneData:
    # Reset the wandb environment variables
    reset_wandb_env()

    seed = worker_data.seed
    group = worker_data.sweep_id
    config = worker_data.config
    worker_num = worker_data.worker_num

    if args.algo == "pgmorl":
        # PGMORL creates its own environments because it requires wrappers
        print(f"Worker {worker_num}: Seed {seed}. Instantiating {args.algo} on {args.env_id}")
        eval_env = mo_gym.make(args.env_id)
        algo = ALGOS[args.algo](
            env_id=args.env_id,
            origin=np.array(args.ref_point),
            wandb_entity=args.wandb_entity,
            **config,
            seed=seed,
            group=group
        )
        print(algo.get_config())

        # Launch the agent training
        print(f"Worker {worker_num}: Seed {seed}. Training agent...")
        algo.train(
            total_timesteps=args.num_timesteps,
            eval_env=eval_env,
            ref_point=np.array(args.ref_point),
            known_pareto_front=None,
            **args.train_hyperparams,
        )

    else:
        print(f"Worker {worker_num}: Seed {seed}. Instantiating {args.algo} on {args.env_id}")
        env = MORecordEpisodeStatistics(mo_gym.make(args.env_id), gamma=config["gamma"])
        eval_env = mo_gym.make(args.env_id)

        algo = ALGOS[args.algo](
            env=env,
            wandb_entity=args.wandb_entity,
            **config,
            seed=seed,
            group=group
        )

        if args.env_id in ENVS_WITH_KNOWN_PARETO_FRONT:
            known_pareto_front = env.unwrapped.pareto_front(gamma=config["gamma"])
        else:
            known_pareto_front = None

        # print(algo.get_config())

        # Launch the agent training
        print(f"Worker {worker_num}: Seed {seed}. Training agent...")
        algo.train(
            eval_env=eval_env,
            ref_point=np.array(args.ref_point),
            known_pareto_front=known_pareto_front,
            **args.train_hyperparams,
        )

        # Get the hypervolume from the wandb run
        hypervolume = wandb.run.summary["eval/hypervolume"]
        print(f"Worker {worker_num}: Seed {seed}. Hypervolume: {hypervolume}")

        return WorkerDoneData(hypervolume=hypervolume)

def main():
    # Get the sweep id
    sweep_run = wandb.init()

    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    with ProcessPoolExecutor(max_workers=args.num_seeds) as executor:
        futures = []
        for num in range(args.num_seeds):
            # print("Spinning up worker {}".format(num))
            seed = seeds[num]
            futures.append(executor.submit(train, WorkerInitData(
                sweep_id=sweep_id,
                seed=seed,
                config=dict(sweep_run.config),
                worker_num=num
            )))

        # Get results from workers
        results = [future.result() for future in futures]

    # Get the hypervolume from the results
    metrics = [result.hypervolume for result in results]

    # Compute the average hypervolume
    average_hypervolume = sum(metrics) / len(metrics)
    print("Average hypervolume: {}".format(average_hypervolume))

    # Log the average hypervolume to the sweep run
    sweep_run.log(dict(hypervolume=average_hypervolume))
    wandb.finish()

args = parse_args()
print(args)

# Create an array of seeds
seeds = [random.randint(0, 1000000) for _ in range(args.num_seeds)]

# Load the sweep config
config_file = os.path.join(os.path.dirname(__file__), "configs", args.config_name)

# Set up the default hyperparameters
with open(config_file) as file:
    sweep_config = yaml.load(file, Loader=yaml.FullLoader)

# Set up the sweep
sweep_id = wandb.sweep(sweep=sweep_config, entity=args.wandb_entity, project=args.project_name)

# Run the sweep agent
wandb.agent(sweep_id, function=main, count=args.sweep_count)
