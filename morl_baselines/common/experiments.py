"""Common experiment utilities."""

import argparse

from morl_baselines.multi_policy.capql.capql import CAPQL
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS, GPIPD
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPILSContinuousAction,
    GPIPDContinuousAction,
)
from morl_baselines.multi_policy.morld.morld import MORLD
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import (
    MPMOQLearning,
)
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from morl_baselines.multi_policy.pcn.pcn import PCN
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL


ALGOS = {
    "pgmorl": PGMORL,
    "envelope": Envelope,
    "gpi_pd_continuous": GPIPDContinuousAction,
    "gpi_pd_discrete": GPIPD,
    "gpi_ls_continuous": GPILSContinuousAction,
    "gpi_ls_discrete": GPILS,
    "capql": CAPQL,
    "mpmoql": MPMOQLearning,
    "pcn": PCN,
    "pql": PQL,
    "ols": MPMOQLearning,
    "gpi-ls": MPMOQLearning,
    "morld": MORLD,
}

ENVS_WITH_KNOWN_PARETO_FRONT = [
    "deep-sea-treasure-concave-v0",
    "deep-sea-treasure-v0",
    "minecart-v0",
    "minecart-deterministic-v0",
    "resource-gathering-v0",
    "fruit-tree-v0",
]


class StoreDict(argparse.Action):
    """Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}

    From RL Baselines3 Zoo.
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        """Init."""
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Convert list of strings to a dict."""
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)
