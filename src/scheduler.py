"""
Experiment runner for MARL (Multi-Agent Reinforcement Learning).

This script:
- Generates experiment grids
- Launches training jobs
- Tracks configurations
- Avoids duplicate runs
- Runs evaluation after training

Designed for reproducible RL benchmarking across MPE and SMAC environments.
"""

import itertools
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
import json

from analyze import load_runs, parse_run_info
from runner import mpe_cooperative_envs

mpe_maps = ["simple_adversary", "simple_crypto", "simple_push", "simple_tag", "simple_spread",
            "simple_reference", "simple_world_comm", "simple_speaker_listener"]

selected_mpe_maps = ["simple_adversary", "simple_push", "simple_spread"]

smac_maps = ["3m", "8m", "13m"]

# Define experimental configurations
# MPE experiments with MAPPO/IPPO
block_1_1 = {
    "environment_name": "mpe",
    "algos": ["ippo", "mappo"],
    "time_steps": 200_000,
    "lrs": [5e-4],
    "seeds": [0, 1, 2, 3, 4],
    "env": mpe_maps,
    "track_carbon": True
}

# MPE experiments with VDN/QMIX
block_1_2 = {
    "environment_name": "mpe",
    "algos": ["vdn", "qmix"],
    "time_steps": 200_000,
    "lrs": [5e-4],
    "seeds": [0, 1, 2, 3, 4],
    "env": mpe_cooperative_envs,
    "track_carbon": True
}

# SMAC experiments with IPPO/MAPPO
block_2_1 = {
    "environment_name": "smac",
    "algos": ["ippo", "mappo"],
    "time_steps": 250_000,
    "lrs": [1e-4],
    "seeds": [0, 1, 2, 3, 4],
    "env": smac_maps,
    "difficulty": [7],
    "track_carbon": True
}

# SMAC experiments with VDN/QMIX
block_2_2 = {
    "environment_name": "smac",
    "algos": ["vdn", "qmix"],
    "time_steps": 250_000,
    "lrs": [1e-4],
    "seeds": [0, 1, 2, 3, 4],
    "env": smac_maps,
    "difficulty": [7],
    "track_carbon": True
}

# SMAC experiments extended with IPPO/MAPPO
block_3_1 = {
    "environment_name": "smac",
    "algos": ["ippo", "mappo"],
    "time_steps": 500_000,
    "lrs": [1e-4],
    "seeds": [0, 1, 2, 3, 4],
    "env": smac_maps,
    "difficulty": [7],
    "track_carbon": True
}

def normalize_config(cfg: dict):
    '''
    Removes non-essential metadata fields from an experiment configuration.

    This is used to ensure fair comparison between experiments by ignoring
    identifiers and filesystem-specific paths.

    :param cfg: Dictionary containing experiment configuration
    :return: Normalized configuration dictionary without metadata keys
    '''
    ignored_keys = {"exp_id", "exp_dir"}

    return {
        k: v
        for k, v in cfg.items()
        if k not in ignored_keys
    }

def configs_match(existing: dict, target: dict):
    '''
    Compares two experiment configurations for equivalence.

    Special rule:
    - 'timesteps' is treated as a minimum threshold instead of strict equality.

    :param existing: Previously executed experiment configuration
    :param target: Desired experiment configuration to match against
    :return: True if configurations are considered equivalent, False otherwise
    '''
    for k, v in target.items():
        if k == "timesteps":
            if existing.get(k, float("inf")) < v:
                return False
        else:
            if existing.get(k) != v:
                return False
    return True

def is_experiment_performed(paths, other_config):
    '''
    Checks whether an experiment with an equivalent configuration
    has already been executed.

    This function prevents redundant computation by comparing the
    normalized configuration of past runs with the target configuration.

    :param paths: List of experiment run directories
    :param other_config: Configuration of the current experiment
    :return: True if experiment already exists, False otherwise
    '''
    target = normalize_config(other_config)

    for p in paths:
        config_path = p.parent.parent.parent / "config.json"
        config = parse_run_info(config_path)

        existing = normalize_config(config)

        if configs_match(existing, target):
            return True

    return False

def run_training(exp_config, path="experiments"):
    '''
    Main pipeline for executing reinforcement learning experiments.

    This function:
    - Generates a full grid of hyperparameter combinations
    - Creates reproducible experiment directories
    - Stores configurations in JSON format
    - Launches training processes via subprocess
    - Handles failed runs and cleans up incomplete experiments
    - Avoids duplicate execution of previously completed experiments

    :param exp_config: Dictionary containing experiment block configuration
    :param path: Root directory where experiments are stored
    :return: None
    '''
    base_dir = Path(path)
    base_dir.mkdir(exist_ok=True)

    paths = load_runs(path)

    failed_runs = []

    grid = list(itertools.product(
        exp_config["algos"],
        exp_config["env"],
        exp_config.get("difficulty", [None]),
        exp_config["lrs"],
        exp_config["seeds"],
    ))

    for algo, env, difficulty, lr, seed in grid:

        exp_id = str(uuid.uuid4())[:8]
        exp_dir = base_dir / f"run_{exp_id}"

        config = {
            "exp_id": exp_id,
            "environment_name": exp_config["environment_name"],
            "env": env,
            "algo": algo,
            "lr": lr,
            "seed": seed,
            "checkpoint_freq": 1000,
            "timesteps": exp_config["time_steps"],
            "workers": 1,
            "local_mode": True,
            "exp_dir": str(exp_dir),
            "track_carbon": exp_config["track_carbon"],
        }

        if exp_config["environment_name"] == "smac" and difficulty is not None:
            config["difficulty"] = difficulty

        restore_path = exp_config.get("restore_path", None)
        if restore_path is not None:
            config["restore_path"] = exp_config["restore_path"]

        if is_experiment_performed(paths, config):
            continue

        exp_dir.mkdir()

        config_path = exp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        print("Launching:", config)

        try:
            subprocess.run(
                [sys.executable, "runner.py", str(config_path), "train"],
                check=True
            )

        except subprocess.CalledProcessError as e:
            print(f"Run failed for config {config_path}")
            print(f"Return code: {e.returncode}")

            # store failed config
            failed_runs.append(config)

            # delete experiment directory
            shutil.rmtree(exp_dir, ignore_errors=True)

            continue  # move to next run

        print(exp_dir)

    # After all runs
    if failed_runs:
        print("=== FAILED RUNS ===")
        for cfg in failed_runs:
            print(cfg)
    else:
        print("All runs completed successfully.")

def run_evaluation(num_eval_seeds=10, path="experiments"):
    '''
    Runs evaluation on trained reinforcement learning experiments.

    For each experiment:
    - Filters relevant environments (e.g., selected MPE maps)
    - Executes evaluation across multiple random seeds
    - Calls external evaluation script via subprocess

    :param num_eval_seeds: Number of evaluation seeds per experiment
    :param path: Directory containing experiment runs
    :return: None
    '''
    paths = load_runs(path)

    for experiment_path in paths:
        if "mpe" in str(experiment_path):

            is_map_valid = False
            for mpe_map in selected_mpe_maps:
                if mpe_map in str(experiment_path):
                    is_map_valid = True
                    break

            if not is_map_valid: continue

        for seed in range(num_eval_seeds):

            print(f"Running experiment={experiment_path}, seed={seed}")

            try:
                subprocess.run(
                    [
                        sys.executable,
                        "evaluate_seed.py",
                        str(experiment_path),
                        str(seed)
                    ],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                pass

if __name__ == "__main__":
    run_training(block_1_1, path="experiments")
    run_training(block_1_2, path="experiments")
    run_training(block_2_1, path="experiments")
    run_training(block_2_2, path="experiments")
    run_training(block_3_1, path="experiments_extended")

    run_evaluation(10, path="experiments")