# Simple env test.
import logging
import os

import aicrowd_helper
import gym
import minerl
from utility.parser import Parser

import coloredlogs
coloredlogs.install(logging.DEBUG)

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4 * 24 * 60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
parser = Parser('performance/',
                allowed_environment=MINERL_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT * 60,
                initial_poll_timeout=600)


# My variables
HDF5_DATA_FILE = "train/data.hdf5"
ACTION_CENTROIDS_FILE = "train/action_centroids.npy"


def main():
    """
    This function will be called for training phase.
    """
    from utils.handle_dataset import store_subset_to_hdf5
    from wrappers.action_wrappers import fit_kmeans

    # For Round1: Skip training, just jump into testing
    return

    # Turn dataset into HDF5
    store_subset_to_hdf5(
        [
            "MineRLTreechopVectorObf-v0",
            "MineRLObtainIronPickaxeVectorObf-v0",
            "MineRLObtainDiamondVectorObf-v0"
        ],
        MINERL_DATA_ROOT,
        HDF5_DATA_FILE
    )

    aicrowd_helper.register_progress(0.50)

    # Fit Kmeans on actions
    # Suuuuuper-elegant argument passing, thanks
    # to the big-brain use of argparse
    kmean_params = [
        "data", HDF5_DATA_FILE,
        "output", ACTION_CENTROIDS_FILE,
        "--n-clusters", "150"
    ]
    fit_kmeans(kmean_params)

    aicrowd_helper.register_progress(1.00)


if __name__ == "__main__":
    main()
