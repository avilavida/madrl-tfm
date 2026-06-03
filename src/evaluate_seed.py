"""
Evaluation module for trained Multi-Agent Reinforcement Learning (MARL) experiments.

This script is responsible for evaluating trained policies stored in experiment
checkpoints and producing quantitative performance metrics.

Key functionalities:
- Loads trained models from the latest checkpoint of an experiment
- Reconstructs the original training environment (MPE / SMAC)
- Restores trained neural network weights and parameters
- Runs deterministic evaluation without exploration noise
- Computes statistical performance metrics over multiple episodes
- Logs results into a centralized CSV file for analysis
- Avoids redundant evaluations by checking previously computed results
- Uses temporary directories for isolated evaluation runs

This module complements the training pipeline by providing a standardized
and reproducible evaluation protocol across all experiments.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import ray

from MARLlib.marllib import marl
from analyze import parse_run_info
from runner import set_seed, share_policy_algos


def run_single_evaluation(experiment_path, eval_seed):
    '''
    Executes evaluation of a trained reinforcement learning experiment.

    This function:
    - Loads the latest trained checkpoint from an experiment
    - Reconstructs the environment and model architecture
    - Restores trained weights and parameters
    - Runs policy evaluation without exploration noise
    - Aggregates performance metrics over multiple episodes
    - Stores results in a CSV file for later analysis

    It also ensures:
    - No duplicate evaluations are executed
    - Temporary evaluation directories are cleaned after execution

    :param experiment_path: Path to the experiment directory containing checkpoints
    :param eval_seed: Random seed used for evaluation (ensures robustness analysis)
    :return: None
    '''

    # Reset Ray state to avoid conflicts from previous runs
    ray.shutdown()

    # File where evaluation results are accumulated across experiments
    csv_path = Path("evaluation_results.csv")

    # Load experiment configuration used during training
    config_path = experiment_path.parent.parent.parent / "config.json"
    # Directory containing model checkpoints
    parent_dir = experiment_path.parent
    params_path = parent_dir / "params.json"

    # Select latest checkpoint based on training iteration number
    checkpoints = list(parent_dir.glob("checkpoint_*"))

    if not checkpoints:
        raise ValueError("No checkpoint folders found")

    checkpoint_dir = max(checkpoints, key=lambda p: int(p.name.split("_")[-1]))
    checkpoint_file = list(checkpoint_dir.glob("checkpoint-*"))[0]

    config = parse_run_info(config_path)

    # Skip evaluation if this experiment-seed pair already exists in results CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        already_done = (
                (df["exp_id"] == config["exp_id"]) &
                (df["eval_seed"] == eval_seed)
        ).any()

        if already_done:
            print(f"Skipping exp={config['exp_id']} seed={eval_seed} (already done)")
            return


    config["env"] = str(config["env"])

    env_kwargs = {}
    if config["environment_name"] == "smac":
        env_kwargs["difficulty"] = config.get("difficulty", "7")

    # Reconstruct environment used during training
    env = marl.make_env(
        environment_name=config["environment_name"],
        map_name=config["env"],
        **env_kwargs
    )

    algo_name = config["algo"]

    algo = getattr(marl.algos, algo_name)(
        hyperparam_source=config["environment_name"],
        lr=config["lr"]
    )

    # Rebuild model architecture and load trained weights
    model = marl.build_model(
        env,
        algo,
        {"core_arch": "mlp", "encode_layer": "128-256"}
    )

    temp_dir = tempfile.mkdtemp(prefix="eval_")

    try:
        set_seed(eval_seed)

        # Run evaluation rollout without exploration noise
        eval_results = algo.fit(
            env,
            model,
            stop={"training_iteration": 1},
            restore_path={
                "model_path": checkpoint_file,
                "params_path": params_path,
            },
            evaluation_interval=1,
            evaluation_num_episodes=100,
            evaluation_config={"explore": False},
            share_policy="group" if algo_name in share_policy_algos else "individual",
            seed=eval_seed,
            local_dir=temp_dir,
        )

        # Extract evaluation metrics from final training iteration
        df = eval_results.results_df
        final_row = df.iloc[-1]

        rewards = final_row.get("evaluation.hist_stats.episode_reward", [])

        # Compute reward variability across episodes
        reward_std = np.std(rewards) if isinstance(rewards, (list, np.ndarray)) else 0.0

        result = {
            "exp_id": config["exp_id"],
            "eval_seed": eval_seed,
            "reward_mean": final_row["evaluation.episode_reward_mean"],
            "reward_std": reward_std,
            "reward_min": final_row["evaluation.episode_reward_min"],
            "reward_max": final_row["evaluation.episode_reward_max"],
        }

        write_header = not csv_path.exists()

        # Append evaluation results to CSV file
        pd.DataFrame([result]).to_csv(
            csv_path,
            mode="a",
            header=write_header,
            index=False
        )

    finally:
        # Remove temporary directory and free resources
        shutil.rmtree(temp_dir, ignore_errors=True)
        ray.shutdown()

if __name__ == "__main__":
    experiment_path = Path(sys.argv[1])
    eval_seed = int(sys.argv[2])

    run_single_evaluation(experiment_path, eval_seed)