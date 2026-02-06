"""
Python script to get the mean accuracies over the different seeds
"""

import ipdb
import os
import sys
import pandas as pd
import numpy as np
import signal
import dotenv
import hydra
from omegaconf import DictConfig

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
ml_path = os.path.dirname(script_dir)

sys.path.append(ml_path)

from src.utils import utils

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

log = utils.get_logger(__name__)

@hydra.main(config_path="configs/", config_name="results_config.yaml", version_base="1.1")
def main(config: DictConfig):

    from src.utils import utils

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    results_dirpath = utils.get_results_path(config=config)

    acc_dfs, acc_vals = [], []

    for run in range(config.n_runs):

        print("-"*50)
        print(f"Retrieving accuracies csv file for run {run}")
        print("-"*50)

        acc_df_path = os.path.join(
            results_dirpath,
            str(run),
            "motion_accuracy_results.csv"
        )

        acc_df = pd.read_csv(acc_df_path)
        acc_dfs.append(acc_df)
        acc_val = acc_df["value"].values
        acc_vals.append(acc_val)

    mean_acc_df = acc_dfs[0].copy()

    mean_acc_df.drop("value",axis=1,inplace=True)
    mean_acc_df["value_mean"] = np.mean(acc_vals,axis=0)
    mean_acc_df["value_median"] = np.median(acc_vals,axis=0)
    mean_acc_df["value_std"] = np.std(acc_vals,axis=0)
    mean_acc_df["value_quantile_01"] = np.quantile(acc_vals,0.1,axis=0)
    mean_acc_df["value_quantile_09"] = np.quantile(acc_vals,0.9,axis=0)

    if config.save_df:
        filepath = os.path.join(results_dirpath,f"mean_acc_df_{config.model_name}_{config.data_encoding}.csv")
        mean_acc_df.to_csv(filepath,index=False)
        print("-"*50)
        print(f"Mean accuracies dataframe saved at: {filepath}")
        print("-"*50)

if __name__ == "__main__":
    main()
