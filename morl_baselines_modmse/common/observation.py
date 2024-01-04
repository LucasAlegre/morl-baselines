import pickle

import gymnasium
import torch as th
from typing import Type, Optional, Any


class Observation:

    """
    This class is an interface used to encapsulate any kind of observation the environment can return.
    Implementing it ensures that all the necessary methods are implemented.
    """

    def __init__(self, item: Optional[Any] = None):
        """
        Initialize the observation.
        Args:
            item: The item to encapsulate. If None, the observation is initialized as None.
        """

        self.item = item
        self.item_dtype = type(item)

    def __repr__(self):
        return self.item.__repr__()

    def __str__(self):
        return self.item.__str__()

    def __eq__(self, other):
        return self.item == other.item

    def __ne__(self, other):
        return self.item != other.item

    def save(self, path):
        """
        Save the observation.
        Args:
            path: The path to save the observation to.
        """
        with open(path, "wb") as f:
            pickle.dump(self.item, f)

    def load(self, path):
        """
        Load the observation.
        Args:
            path: The path to load the observation from.
        """
        with open(path, "rb") as f:
            self.item = pickle.load(f)
            self.item_dtype = type(self.item)

    def to_tensor(self, device=None):
        """
        Convert the observation to a PyTorch tensor.
        Args:
            device: The device to use.
        Returns:
            The observation as a PyTorch tensor.
        """
        if device is None:
            return th.tensor(self.item, device=th.device("cpu"))
        else:
            return th.tensor(self.item, device=device)


class ConversionWrapper(gymnasium.ObservationWrapper):

    """
    This class is used to wrap the observations returned by the environment.
    It is used to ensure that the observations are of the Observation type.
    """

    def __init__(self, env: gymnasium.Env, observation_class: Type[Observation] = Observation, observation_space: gymnasium.Space = None):
        """
        Initialize the wrapper.
        Args:
            env: The environment to wrap.
            observation_class: The class to use for the observations. By default, it is Observation, feel free to implement your own inheriting from it.
            observation_space: The observation space after conversion (if different from the original).
        """
        super().__init__(env)
        self.observation_class = observation_class
        if observation_space is not None:
            self.observation_space = observation_space
        else:
            self.observation_space = env.observation_space

    def observation(self, observation):
        """
        Wrap the observation.
        Args:
            observation: The observation to wrap.
        Returns:
            The wrapped observation.
        """
        return self.observation_class(observation)
