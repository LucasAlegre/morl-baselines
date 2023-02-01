import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb


api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("florianfelten/MORL-Baselines")
METRIC = "charts/eum"
FANCY_METRIC_NAME = "EUM"
ENV = "mo-hopper-v4"

runs = [
    r
    for r in runs
    # Some filters because I ran shitloads of experiments
    if r.config["env_name"] == ENV and r.summary["global_step"] == 8050000 and "eum_samples" in r.config.keys()
]


hypervolumes_vanilla = []
hypervolumes_sb_psa = []
for run in runs:
    hypervolume_df = run.history(samples=1000000000000, keys=[METRIC], x_axis="global_step").dropna()
    hypervolume_df = hypervolume_df.rename(columns={METRIC: FANCY_METRIC_NAME, "global_step": "Timesteps"})
    hypervolume_df.set_index("Timesteps")
    if run.config["shared_buffer"]:
        hypervolumes_sb_psa.append(hypervolume_df)
    else:
        hypervolumes_vanilla.append(hypervolume_df)


vanilla_df = pd.concat(hypervolumes_vanilla, axis=0).assign(variant="MORL/D vanilla")
sb_psa_df = pd.concat(hypervolumes_sb_psa, axis=0).assign(variant="MORL/D-SB+PSA")

concatenated = pd.concat(
    [
        vanilla_df,
        sb_psa_df,
    ]
)

sns.set_style("darkgrid")
sns.color_palette("deep")
# plt.ylim(0, 800)
l = sns.lineplot(data=concatenated, x="Timesteps", y=FANCY_METRIC_NAME, hue="variant", alpha=0.6, errorbar=("ci", 95), seed=42)
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
plt.savefig(f"{FANCY_METRIC_NAME}_{ENV}.png", dpi=600)
