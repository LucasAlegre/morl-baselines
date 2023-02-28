import json
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

from morl_baselines.common.pareto import ParetoArchive
from morl_baselines.common.performance_indicators import igd


def read_table_file(filename):
    """Reads a table from wandb as json and return a 2D list containing a pareto front."""
    with open(filename) as f:
        data = json.load(f)
    return data["data"]


api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("florianfelten/MORL-Baselines")
ENV = "mo-hopper-v4"

runs = [
    r for r in runs if r.config["env_name"] == ENV and r.summary["global_step"] == 8050000 and "eum_samples" in r.config.keys()
]

last_fronts = []
fronts_vanilla = []
fronts_sb_psa = []
for run in runs:
    df = run.history(samples=1000000000000, keys=["charts/pareto_front"], x_axis="global_step").dropna()
    df = df.rename(columns={"charts/pareto_front": "PF", "global_step": "Timesteps"})
    df.set_index("Timesteps")

    # Fronts are stored in json files that we need to download
    filenames = [d.get("path") for d in df["PF"]]
    for filename in filenames:
        try:
            run.file(filename).download(replace=False)
        except Exception:
            pass

    fronts = [read_table_file(filename) for filename in filenames]
    df["PF"] = fronts
    last_fronts.append(fronts[-1])
    if run.config["shared_buffer"]:
        fronts_sb_psa.append(df)
    else:
        fronts_vanilla.append(df)

merged_front = ParetoArchive()
for front in last_fronts:
    for point in front:
        merged_front.add(candidate=None, evaluation=np.array(point))


def calc_igd(ref_front: List[np.ndarray], front: List[np.ndarray]):
    return igd(known_front=ref_front, current_estimate=front)


for df in fronts_vanilla:
    df["IGD"] = df["PF"].apply(lambda x: calc_igd(merged_front.evaluations, x))
for df in fronts_sb_psa:
    df["IGD"] = df["PF"].apply(lambda x: calc_igd(merged_front.evaluations, x))

vanilla_df = pd.concat(fronts_vanilla, axis=0).assign(variant="MORL/D vanilla")
sb_psa_df = pd.concat(fronts_sb_psa, axis=0).assign(variant="MORL/D-SB+PSA")

concatenated = pd.concat(
    [
        vanilla_df,
        sb_psa_df,
    ]
)


sns.set_style("darkgrid")
sns.color_palette("deep")
l = sns.lineplot(data=concatenated, x="Timesteps", y="IGD", hue="variant", alpha=0.6, errorbar=("ci", 95), seed=42)
sns.despine()
sns.move_legend(
    l,
    "lower center",
    bbox_to_anchor=(0.5, 1),
    ncol=2,
    title=None,
    frameon=False,
)
plt.tight_layout()
plt.savefig("igd_hopper.png", dpi=600)
