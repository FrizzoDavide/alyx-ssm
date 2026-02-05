"""
Python script to launch the training experiment
"""

import ipdb
import glob
import os
import signal
from pathlib import Path

import dotenv
import hydra
from omegaconf import DictConfig

from src.utils import utils

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

log = utils.get_logger(__name__)


@hydra.main(config_path="configs/", config_name="test_config.yaml", version_base="1.1")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.test import test
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    #NOTE: Automatic computation of the model checkpoint path

    ml_path = Path(Path.home() / "alyx-ssm" / "machine_learning")

    outputs_path = os.path.join(ml_path,"multirun") if config.multirun else os.path.join(ml_path,"outputs")

    model_day_dirpath = utils.get_most_recent_dir(outputs_path)
    model_hour_dirpath = utils.get_most_recent_dir(model_day_dirpath)

    if config.multirun:

        model_checkpoint_dirpath = utils.generate_path(
            basepath = model_hour_dirpath,
            folders = [
                str(config.run_number),
                "checkpoints"
            ]
        )

    else:

        model_checkpoint_dirpath = utils.generate_path(
            basepath = model_hour_dirpath,
            folders = ["checkpoints"]
        )

    file_strategy = os.path.join(model_checkpoint_dirpath,f"*{config.strategy}*.ckpt")
    path_to_model = glob.glob(file_strategy)[0]

    print("-"*50)
    print(f"Test experiment on {path_to_model} model checkpoint")
    print("-"*50)

    # Test model
    return test(config, path_to_model)


def trap_signals_to_handle_kills_gracefully():
    def handle_kill(signal_number, _frame):
        log.warning("getting killed")
        try:
            import wandb
            wandb.mark_preempting()
            wandb.finish(exit_code=-1)
        except Exception as e:
            log.error("an error occurred during training abort:")
            log.error(e)

        signal_name = {int(s): s for s in signal.Signals}[signal_number].name
        log.warning(f"aborting because of {signal_name} signal")
        raise SystemExit(f"aborting because of {signal_name} signal")

    signal.signal(signal.SIGINT, handle_kill)
    signal.signal(signal.SIGTERM, handle_kill)


if __name__ == "__main__":
    trap_signals_to_handle_kills_gracefully()
    main()
