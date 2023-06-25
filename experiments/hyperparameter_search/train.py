def create_algorithm(args, config, seed, group):
    return ALGOS[args.algo](
        env=mo_gym.make(args.env_id),
        wandb_entity=args.wandb_entity,
        **config,
        seed=seed,
        group=group
    )

def get_known_pareto_front(env_id, config):
    if env_id in ENVS_WITH_KNOWN_PARETO_FRONT:
        return MORecordEpisodeStatistics(mo_gym.make(env_id), gamma=config["gamma"]).unwrapped.pareto_front(gamma=config["gamma"])
    return None

def train_worker(worker_data: WorkerInitData) -> WorkerDoneData:
    reset_wandb_env()

    args = worker_data.config.pop("args")
    seed, group, config, worker_num = worker_data.seed, worker_data.sweep_id, worker_data.config, worker_data.worker_num

    print(f"Worker {worker_num}: Seed {seed}. Instantiating {args.algo} on {args.env_id}")
    algo = create_algorithm(args, config, seed, group)
    eval_env = mo_gym.make(args.env_id)
    known_pareto_front = get_known_pareto_front(args.env_id, config)

    print(f"Worker {worker_num}: Seed {seed}. Training agent...")
    algo.train(
        eval_env=eval_env,
        ref_point=np.array(args.ref_point),
        known_pareto_front=known_pareto_front,
        **args.train_hyperparams,
    )

    hypervolume, igd = wandb.run.summary["eval/hypervolume"], None
    print(f"Worker {worker_num}: Seed {seed}. Hypervolume: {hypervolume}")

    if known_pareto_front:
        igd = wandb.run.summary["eval/igd"]
        print(f"Worker {worker_num}: Seed {seed}. IGD: {igd}")

    return WorkerDoneData(hypervolume=hypervolume, igd=igd)






def create_algo(worker_data: WorkerInitData, env=None):
    return ALGOS[args.algo](
        env=env,
        env_id=args.env_id,
        origin=np.array(args.ref_point),
        wandb_entity=args.wandb_entity,
        **worker_data.config,
        seed=worker_data.seed,
        group=worker_data.sweep_id
    )


def create_environment(env_id: str, env_type: str):
    if env_type == "eval":
        return mo_gym.make(env_id)
    elif env_type == "train":
        return MORecordEpisodeStatistics(mo_gym.make(env_id), gamma=config["gamma"])


def train(worker_data: WorkerInitData) -> WorkerDoneData:
    reset_wandb_env()
    print(f"Worker {worker_data.worker_num}: Seed {worker_data.seed}. Instantiating {args.algo} on {args.env_id}")

    algo_kwargs = {
        "eval_env": create_environment(args.env_id, "eval"),
        "ref_point": np.array(args.ref_point),
        **args.train_hyperparams,
    }

    if args.algo == "pgmorl":
        env = None
    else:
        env = create_environment(args.env_id, "train")
        if args.env_id in ENVS_WITH_KNOWN_PARETO_FRONT:
            algo_kwargs["known_pareto_front"] = env.unwrapped.pareto_front(gamma=worker_data.config["gamma"])
        else:
            algo_kwargs["known_pareto_front"] = None

    algo = create_algo(worker_data, env)

    print(f"Worker {worker_data.worker_num}: Seed {worker_data.seed}. Training agent...")
    algo.train(**algo_kwargs)

    hypervolume = wandb.run.summary["eval/hypervolume"]
    print(f"Worker {worker_data.worker_num}: Seed {worker_data.seed}. Hypervolume: {hypervolume}")

    igd = wandb.run.summary["eval/igd"] if algo_kwargs["known_pareto_front"] else None
    if igd:
        print(f"Worker {worker_data.worker_num}: Seed {worker_data.seed}. IGD: {igd}")

    return WorkerDoneData(hypervolume=hypervolume, igd=igd)
