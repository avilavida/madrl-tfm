"""
Multi-Agent Reinforcement Learning (MARL) experiment execution module.

This script is responsible for running a single training experiment
in a reproducible and fully configured environment.

Key responsibilities:
- Initialize deterministic execution (seeds for reproducibility)
- Create MARL environments (MPE and SMAC benchmarks)
- Instantiate reinforcement learning algorithms (IPPO, MAPPO, QMIX, VDN)
- Build neural network models using MARLlib
- Execute training using Ray distributed backend
- Optionally track environmental impact using CodeCarbon (CO₂ emissions)
- Save results and checkpoints in structured experiment directories
- Ensure clean termination of distributed resources

This module is designed to be executed from a higher-level experiment
manager that handles hyperparameter grids and batch execution.
"""

import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ray
import torch
from codecarbon import EmissionsTracker

from MARLlib.marllib import marl

from analyze import parse_run_info, load_runs

mpe_cooperative_envs = ["simple_spread", "simple_reference"]
share_policy_algos = ["qmix", "mappo"]

def set_seed(seed):
    '''
    Sets global random seeds to ensure reproducibility of experiments.

    This function fixes randomness across:
    - Python random module
    - NumPy
    - PyTorch (CPU and GPU)
    - CUDA backend behavior

    It also enforces deterministic computation in CUDA operations.

    :param seed: Integer seed value used for reproducibility
    :return: None
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiment(config):
    '''
    Executes a single reinforcement learning experiment.

    This function is responsible for:
    - Setting reproducibility seed
    - Creating the multi-agent environment (MPE or SMAC)
    - Initializing the selected MARL algorithm
    - Building the neural network model
    - Optionally tracking carbon emissions (CodeCarbon)
    - Training the agent using MARLlib
    - Saving results and checkpoints
    - Cleaning up Ray execution environment

    :param config: Dictionary containing experiment configuration parameters
    :return: None
    '''
    seed = config.get("seed", 42)

    set_seed(seed)

    env_kwargs = {}

    # Add SMAC-specific environment parameters (difficulty level)
    if config["environment_name"] == "smac":
        env_kwargs["difficulty"] = config.get("difficulty", "7")

    env = marl.make_env(
        environment_name=config['environment_name'],
        map_name=config['env'],
        **env_kwargs
    )

    algo_name = config["algo"]
    # Initialize selected MARL algorithm (e.g., IPPO, MAPPO, QMIX, VDN)
    algo = getattr(marl.algos, algo_name)(
        hyperparam_source=config["environment_name"],
        lr=config["lr"],
    )

    # Build neural network architecture for multi-agent training
    model = marl.build_model(
        env,
        algo,
        {"core_arch": "mlp", "encode_layer": "128-256"}
    )

    tracker: Optional[EmissionsTracker] = None
    track_carbon = config.get("track_carbon", False)

    # Optional CO2 emission tracking for sustainability analysis
    if track_carbon:
        tracker = EmissionsTracker(
            output_dir=config["exp_dir"],
            output_file="emissions.csv",
            log_level="error"
        )
        tracker.start()

    try:
        # Train agent until reaching total timesteps budget
        results = algo.fit(
            env,
            model,
            stop={"timesteps_total": config["timesteps"]},
            local_mode=config.get("local_mode", True),
            num_gpus=0,
            num_workers=config.get("workers", 1),
            share_policy = "group" if algo_name in share_policy_algos else "individual",
            checkpoint_freq=config.get("checkpoint_freq", 0),
            evaluation_interval=1,
            evaluation_num_episodes=50,
            evaluation_config={"explore": False}, # Evaluation runs without exploration noise for fair performance measurement
            seed=seed,
            local_dir=config["exp_dir"],
        )
    finally:
        if tracker is not None:
            emissions = tracker.stop()
            print(f"Emissions for this run: {emissions} kg CO2")
        # Ensure Ray is properly shut down after execution
        ray.shutdown()

    # Force process termination to avoid Ray hanging states after training
    os._exit(0)

if __name__ == "__main__":
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)

    run_experiment(config)