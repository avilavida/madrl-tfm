"""
Analysis and visualization module for Multi-Agent Reinforcement Learning (MARL) experiments.

This script is responsible for post-processing and analyzing experimental results
generated during training and evaluation phases.

It provides a complete analysis pipeline that includes:

- Loading and parsing experiment logs from disk
- Aggregating training reward curves and carbon emission data
- Preprocessing and smoothing learning curves
- Generating visualizations for:
    * Learning performance over time
    * Final evaluation performance distribution
    * Algorithm comparison via barplots and boxplots
    * Carbon footprint (CO₂ emissions) analysis
- Producing summary tables combining performance and sustainability metrics

This module enables a comprehensive comparison of MARL algorithms
(IPPO, MAPPO, VDN, QMIX) across different environments (MPE, SMAC),
supporting both performance and environmental impact analysis.

The outputs are used for figures and results presented in the final TFM report.
"""

import json
from pathlib import Path
from pprint import pprint

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

base_dir = Path(__file__).parent.parent

def load_runs(base="experiments"):
    '''
    Loads all experiment run outputs from disk.

    It recursively searches for training progress files
    generated during experiments.

    :param base: Root directory containing experiment folders
    :return: List of paths to progress.csv files
    '''
    paths = list(Path(base).rglob("progress.csv"))
    return paths

def parse_run_info(config_path):
    '''
    Loads and parses a JSON configuration file for a single experiment.

    :param config_path: Path to config.json file
    :return: Dictionary containing experiment configuration
    '''
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def aggregate_runs(paths):
    '''
    Aggregates raw experiment outputs into structured datasets.

    This function extracts:
    - Training reward curves
    - Carbon emission logs
    - Experiment metadata

    It aligns all runs into a unified format for analysis and plotting.

    :param paths: List of progress.csv files from experiments
    :return: Tuple containing:
             (metadata dataframe, rewards dataframe, emissions dataframe)
    '''
    data_meta = []
    data_rewards = []
    data_emissions = []

    for p in paths:
        df_rewards = pd.read_csv(p)

        run_id = p.parent.parent.parent.name.replace("run_", "")
        config_path = p.parent.parent.parent / "config.json"
        config = parse_run_info(config_path)

        df_rewards = df_rewards[["timesteps_total", "evaluation/episode_reward_mean"]].dropna()

        emissions_path = p.parent.parent.parent / "emissions.csv"
        df_emissions = pd.read_csv(emissions_path)
        df_emissions = df_emissions[["duration", "emissions", "water_consumed"]].dropna()

        df_rewards["algo"] = config['algo']
        df_rewards["environment_name"] = config['environment_name']
        df_rewards["env"] = config['env']
        df_rewards["run_id"] = run_id

        df_emissions["algo"] = config['algo']
        df_emissions["environment_name"] = config['environment_name']
        df_emissions["env"] = config['env']
        df_emissions["run_id"] = run_id

        data_meta.append({
            "run_id": run_id,
            "algo": config["algo"],
            "environment_name": config["environment_name"],
            "env": config["env"],
        })
        data_rewards.append(df_rewards)
        data_emissions.append(df_emissions)

    return pd.DataFrame(data_meta), pd.concat(data_rewards), pd.concat(data_emissions)

def preprocess_rewards(df, max_time_steps=None):
    '''
    Preprocesses reward data for visualization.

    Steps:
    - Optionally filters by maximum timesteps
    - Discretizes timesteps for smoother aggregation

    :param df: Raw reward dataframe
    :param max_time_steps: Optional upper limit for timesteps
    :return: Processed dataframe
    '''
    df = df.copy()

    if max_time_steps is not None:
        df = df[df["timesteps_total"] <= max_time_steps]

    df["timesteps_total"] = (df["timesteps_total"] // 5000) * 5000
    return df

def plot_learning_curves(base="experiments", df=None, save_plots=False):
    '''
    Plots training learning curves for all algorithms and environments.

    Displays:
    - Mean episodic reward over training
    - Standard deviation as confidence band

    This allows comparison of learning stability and convergence speed
    across different MARL algorithms.

    :param base: Experiment root directory
    :param df: Processed rewards dataframe
    :param save_plots: Whether to save figures to disk
    :return: None
    '''
    algos = sorted(df["algo"].unique())

    cmap = plt.get_cmap("tab10")

    algo_colors = {
        algo: cmap(i % cmap.N)
        for i, algo in enumerate(algos)
    }

    for env_name in df["environment_name"].unique():
        full_output_dir = base_dir / "results" / env_name / "learning_curves"

        full_output_dir.mkdir(parents=True, exist_ok=True)

        for env in df["env"].unique():
            plt.figure(env)
            env_df = df[df["env"] == env]

            for algo in sorted(env_df["algo"].unique()):
                sub = env_df[env_df["algo"] == algo]

                rewards = sub.groupby("timesteps_total")["evaluation/episode_reward_mean"]
                mean = rewards.mean()
                std = rewards.std()

                color = algo_colors[algo]

                plt.plot(
                    mean.index,
                    mean.values,
                    label=algo,
                    color=color
                )
                plt.fill_between(
                    mean.index,
                    mean - std,
                    mean + std,
                    alpha=0.2,
                    color=color
                )

            plt.title(f"Learning Curves")
            plt.xlabel("Training Timesteps")
            plt.ylabel("Episodic Reward")
            plt.legend()

            if save_plots:
                if base == "experiments_2":
                    filename = f"learning_curves_extended_{env}.png"
                else:
                    filename = f"learning_curves_{env}.png"
                filepath = full_output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches="tight")

            plt.show()
            plt.close()

def plot_barplot(base="experiments", df=None, value_col=None, title_prefix="", y_label="", save_plots=False, output_dir=None):
    '''
    Generates bar plots comparing algorithm performance.

    Aggregates results per algorithm and scenario and visualizes:
    - Mean performance metrics
    - Standard deviation (variability across runs)

    Useful for high-level comparison of MARL algorithms.

    :return: None
    '''
    algos = sorted(df["algo"].unique())

    cmap = plt.get_cmap("tab10")

    algo_colors = {
        algo: cmap(i % cmap.N)
        for i, algo in enumerate(algos)
    }

    for env_name in df["environment_name"].unique():
        full_output_dir = base_dir / "results" / env_name / "barplots"
        if output_dir is not None:
            full_output_dir = full_output_dir / output_dir
        full_output_dir.mkdir(parents=True, exist_ok=True)

        for scenario, df_env in df.groupby("env"):
            plt.figure(scenario)

            mean_df = (
                df_env
                .groupby("algo")[value_col]
                .mean()
                .reset_index()
            )

            sns.barplot(
                x="algo",
                y=value_col,
                hue="algo",
                data=mean_df,
                estimator="mean",
                errorbar="sd",
                palette=algo_colors,
                legend=False
            )

            plt.title(f"{title_prefix}")
            plt.xlabel("Algorithm")
            plt.ylabel(y_label)

            plt.tight_layout()

            if save_plots:
                filename = title_prefix.lower().replace(" ", "_")
                if base == "experiments_2":
                    filename = f"{filename}_extended_{scenario}.png"
                else:
                    filename = f"{filename}_{scenario}.png"
                filepath = full_output_dir / filename

                plt.savefig(
                    filepath,
                    dpi=300,
                    bbox_inches="tight"
                )

            plt.show()
            plt.close()

def plot_boxplot(base="experiments", df=None, value_col=None, title_prefix="", save_plots=False, output_dir=None):
    '''
    Generates boxplots to visualize distribution of results.

    Highlights:
    - Variability between runs
    - Robustness of each algorithm

    :return: None
    '''
    algo_order = sorted(df["algo"].unique())

    for env_name in df["environment_name"].unique():
        full_output_dir = base_dir / "results" / env_name / "boxplots"
        if output_dir is not None:
            full_output_dir = full_output_dir / output_dir
        full_output_dir.mkdir(parents=True, exist_ok=True)

        for scenario, df_env in df.groupby("env"):

            plt.figure(scenario)

            sns.boxplot(
                x="algo",
                y=value_col,
                data=df_env,
                order = [a for a in algo_order if a in df_env["algo"].unique()]
            )

            plt.title(f"{title_prefix}")
            plt.xlabel("Algorithm")

            y_label = value_col.replace("_", " ").title()
            plt.ylabel(y_label)
            plt.tight_layout()

            if save_plots:
                filename = title_prefix.lower().replace(" ", "_")
                if base == "experiments_2":
                    filename = f"{filename}_extended_{scenario}.png"
                else:
                    filename = f"{filename}_{scenario}.png"
                filepath = full_output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches="tight")

            plt.show()
            plt.close()

def plot_final_reward(base="experiments", df=None, save_plots=False):
    '''
    Visualizes final evaluation performance of each algorithm.

    Uses boxplots to show distribution of final episode rewards.

    :return: None
    '''
    plot_boxplot(
        base,
        df,
        value_col="reward_mean",
        title_prefix="Final Reward",
        save_plots=save_plots,
        output_dir="final_rewards"
    )

def plot_emissions(base, df, save_plots=False):
    '''
    Visualizes carbon emissions generated during training.

    Compares environmental cost of different MARL algorithms.

    :return: None
    '''
    plot_barplot(
        base,
        df,
        value_col="emissions",
        title_prefix="Carbon Emissions",
        y_label="Mean CO₂ emissions (kgCO₂)",
        save_plots=save_plots,
        output_dir="emissions"
    )

def print_summary_table(df_eval, df_emissions):
    '''
    Prints a summarized table combining:
    - Evaluation performance metrics
    - Carbon emissions statistics

    Aggregates results per algorithm and environment.

    :return: None
    '''
    df_eval = df_eval.drop(['env', 'environment_name', 'algo'], axis=1)
    df_emissions = df_emissions.drop(['duration', 'water_consumed'], axis=1)
    df_summary = pd.merge(df_eval, df_emissions, on="run_id", how="inner")
    df_summary = (
        df_summary.groupby(["algo", "env", "environment_name"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(by=["environment_name", "env", "algo"], ascending=False)
    )

    pprint(df_summary)

def analyse(base="experiments", env_name=None, max_time_steps=None, save_plots=False):
    '''
    Main analysis pipeline for reinforcement learning experiments.

    This function orchestrates the full post-processing workflow:

    1. Loads experiment runs from disk
    2. Aggregates training and emission data
    3. Preprocesses reward curves
    4. Generates learning curves and emission plots
    5. Loads evaluation results
    6. Prints summary statistics
    7. Produces final performance visualizations

    :param base: Root directory of experiments
    :param env_name: Specific environment to analyze
    :param max_time_steps: Optional truncation of training curves
    :param save_plots: Whether to save generated figures
    :return: None
    '''
    paths = load_runs(base)

    df_meta, df_rewards, df_emissions = aggregate_runs(paths)
    df_rewards = preprocess_rewards(df_rewards, max_time_steps)

    plot_learning_curves(base=base, df=df_rewards[df_rewards['environment_name'] == env_name], save_plots=save_plots)
    plot_emissions(base=base, df=df_emissions[df_emissions['environment_name'] == env_name], save_plots=save_plots)

    df_eval = pd.read_csv(Path(base_dir / "results" / "evaluation_results.csv"))

    df_eval = df_eval.merge(
        df_meta,
        left_on="exp_id",
        right_on="run_id",
        how="inner"
    )
    print_summary_table(df_eval, df_emissions)
    df_eval = df_eval.drop(columns=["run_id"])
    plot_final_reward(base=base, df=df_eval[df_eval['environment_name'] == env_name], save_plots=save_plots)

if __name__ == "__main__":
    analyse(base="experiments", env_name='smac', max_time_steps=250_000, save_plots=False)
