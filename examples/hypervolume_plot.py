import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb


api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("florianfelten/MORL-Baselines")
runs = [r for r in runs if r.config["env_name"] == "mo-halfcheetah-v4" and r.summary["global_step"] == 6050000]
summary_list = []
config_list = []
hypervolumes_vanilla = []
hypervolumes_sb_psa = []
name_list = []
for run in runs:
    hypervolume_df = run.history(samples=1000000000000, keys=["charts/hypervolume"], x_axis="global_step").dropna()
    hypervolume_df = hypervolume_df.rename(columns={"charts/hypervolume": "Hypervolume", "global_step": "Timesteps"})
    if run.config["shared_buffer"]:
        hypervolumes_sb_psa.append(hypervolume_df)
    else:
        hypervolumes_vanilla.append(hypervolume_df)


vanilla_dfs = [df.set_index("Timesteps") for df in hypervolumes_vanilla]
sb_psa_dfs = [df.set_index("Timesteps") for df in hypervolumes_sb_psa]

vanilla_df = pd.concat(vanilla_dfs, axis=0).assign(variant="MORL/D vanilla")
sb_psa_df = pd.concat(sb_psa_dfs, axis=0).assign(variant="MORL/D-SB+PSA")

concatenated = pd.concat(
    [
        vanilla_df,
        sb_psa_df,
    ]
)


sns.set(style="darkgrid")
sns.lineplot(
    data=concatenated, x="Timesteps", y="Hypervolume", hue="variant", palette=["red", "blue"], alpha=0.6, errorbar="ci"
)
plt.savefig("hypervolume_cheetah.png", dpi=600)
