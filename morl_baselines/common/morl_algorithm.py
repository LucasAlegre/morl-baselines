"""MORL algorithm base classes."""
from abc import ABC, abstractmethod
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from mo_gymnasium import eval_mo, eval_mo_reward_conditioned
from torch.utils.tensorboard import SummaryWriter


class MOPolicy(ABC):
    """An MORL policy.

    It has an underlying learning structure which can be:
    - used to get a greedy action via eval()
    - updated using some experiences via update()

    Note that the learning structure can embed multiple policies (for example using a Conditioned Network).
    In this case, eval() requires a weight vector as input.
    """

    def __init__(self, id: Optional[int] = None, device: Union[th.device, str] = "auto") -> None:
        """Initializes the policy.

        Args:
            id: The id of the policy
            device: The device to use for the tensors
        """
        self.id = id
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device == "auto" else device
        self.global_step = 0

    @abstractmethod
    def eval(self, obs: np.ndarray, w: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        """Gives the best action for the given observation.

        Args:
            obs (np.array): Observation
            w (optional np.array): weight for scalarization

        Returns:
            np.array or int: Action
        """

    def __report(
        self,
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        discounted_vec_return,
        writer: SummaryWriter,
    ):
        """Writes the data to wandb summary."""
        if self.id is None:
            idstr = ""
        else:
            idstr = f"_{self.id}"

        writer.add_scalar(f"eval{idstr}/scalarized_return", scalarized_return, self.global_step)
        writer.add_scalar(
            f"eval{idstr}/scalarized_discounted_return",
            scalarized_discounted_return,
            self.global_step,
        )
        for i in range(vec_return.shape[0]):
            writer.add_scalar(f"eval{idstr}/vec_{i}", vec_return[i], self.global_step)
            writer.add_scalar(
                f"eval{idstr}/discounted_vec_{i}",
                discounted_vec_return[i],
                self.global_step,
            )
        log_dict = {
            "train/step": self.global_step  
            "train/scalarized_return": scalarized_return, 
            "train/scalarized_discounted_return": scalarized_discounted_return, 
            "train/vec_return": vec_return, 
            "train/discounted_vec_return": discounted_vec_return
        }
        wandb.log(log_dict)

    def policy_eval(
        self,
        eval_env,
        scalarization=np.dot,
        weights: Optional[np.ndarray] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        """Runs a policy evaluation (typically on one episode) on eval_env and logs some metrics using writer.

        Args:
            eval_env: evaluation environment
            scalarization: scalarization function
            weights: weights to use in the evaluation
            writer: wandb writer

        Returns:
             a tuple containing the evaluations
        """
        (
            scalarized_reward,
            scalarized_discounted_reward,
            vec_reward,
            discounted_vec_reward,
        ) = eval_mo(self, eval_env, scalarization=scalarization, w=weights)

        if writer is not None:
            self.__report(
                scalarized_reward,
                scalarized_discounted_reward,
                vec_reward,
                discounted_vec_reward,
                writer,
            )

        return scalarized_reward, scalarized_discounted_reward, vec_reward, discounted_vec_reward

    def policy_eval_esr(
        self,
        eval_env,
        scalarization,
        weights: Optional[np.ndarray] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        """Runs a policy evaluation (typically on one episode) on eval_env and logs some metrics using writer.

        Args:
            eval_env: evaluation environment
            scalarization: scalarization function
            weights: weights to use in the evaluation
            writer: wandb writer

        Returns:
             a tuple containing the evaluations
        """
        (
            scalarized_reward,
            scalarized_discounted_reward,
            vec_reward,
            discounted_vec_reward,
        ) = eval_mo_reward_conditioned(self, eval_env, scalarization, weights)

        if writer is not None:
            self.__report(
                scalarized_reward,
                scalarized_discounted_reward,
                vec_reward,
                discounted_vec_reward,
                writer,
            )

        return scalarized_reward, scalarized_discounted_reward, vec_reward, discounted_vec_reward

    @abstractmethod
    def update(self) -> None:
        """Update algorithm's parameters (e.g. using experiences from the buffer)."""


class MOAgent(ABC):
    """An MORL Agent, can contain one or multiple MOPolicies. Contains helpers to extract features from the environment, setup logging etc."""

    def __init__(self, env: Optional[gym.Env], device: Union[th.device, str] = "auto") -> None:
        """Initializes the agent.

        Args:
            env: (gym.Env): The environment
            device: (str): The device to use for training. Can be "auto", "cpu" or "cuda".
        """
        self.extract_env_info(env)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device == "auto" else device

        self.global_step = 0
        self.num_episodes = 0

    def extract_env_info(self, env: Optional[gym.Env]) -> None:
        """Extracts all the features of the environment: observation space, action space, ...

        Args:
            env (gym.Env): The environment
        """
        # Sometimes, the environment is not instantiated at the moment the MORL algorithms is being instantiated.
        # So env can be None. It is the responsibility of the implemented MORLAlgorithm to call this method in those cases
        if env is not None:
            self.env = env
            if isinstance(self.env.observation_space, spaces.Discrete):
                self.observation_shape = (1,)
                self.observation_dim = self.env.observation_space.n
            else:
                self.observation_shape = self.env.observation_space.shape
                self.observation_dim = self.env.observation_space.shape[0]

            self.action_space = env.action_space
            if isinstance(self.env.action_space, (spaces.Discrete, spaces.MultiBinary)):
                self.action_shape = (1,)
                self.action_dim = self.env.action_space.n
            else:
                self.action_shape = self.env.action_space.shape
                self.action_dim = self.env.action_space.shape[0]
            self.reward_dim = self.env.reward_space.shape[0]

    @abstractmethod
    def get_config(self) -> dict:
        """Generates dictionary of the algorithm parameters configuration.

        Returns:
            dict: Config
        """

    def setup_wandb(self, project_name: str, experiment_name: str) -> None:
        """Initializes the wandb writer.

        Args:
            project_name: name of the wandb project. Usually MORL-Baselines.
            experiment_name: name of the wandb experiment. Usually the algorithm name.

        Returns:
            None
        """
        self.experiment_name = experiment_name
        import wandb

        wandb.init(
            project=project_name,
            sync_tensorboard=True,
            config=self.get_config(),
            name=self.experiment_name,
            monitor_gym=False,
            save_code=True,
        )
        self.writer = SummaryWriter(f"/tmp/{self.experiment_name}")
        # The default "step" of wandb is not the actual time step (gloabl_step) of the MDP
        wandb.define_metric("*", step_metric="global_step")
        #to save the step log in the folder "train"
        wandb.define_metric("train/step")
        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="train/step")

    def close_wandb(self) -> None:
        """Closes the wandb writer and finishes the run."""
        import wandb

        self.writer.close()
        wandb.finish()
