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
HDF5_DATA_FILE_FRAMESKIPPED = "train/data_frameskipped.hdf5"
ACTION_CENTROIDS_FILE = "train/action_centroids.npy"
TRAINED_MODEL_PATH = "train/trained_model.th"


def main():
    """
    This function will be called for training phase.
    """
    from utils.handle_dataset import store_subset_to_hdf5, remove_frameskipped_samples
    from wrappers.action_wrappers import fit_kmeans, update_hdf5_with_centroids
    from train_bc_lstm import main as main_train_bc
    from train_bc_lstm import parser as train_bc_parser

    # For Round1: Skip training, just jump into testing
    print("--- WARNING --- Training has been disabled for Round1 submission --- ")
    print("---             Remove `return` from train.py:main() to run training ---")
    return
    
    aicrowd_helper.training_start()
    # Turn dataset into HDF5
    store_subset_to_hdf5_params = [
        MINERL_DATA_ROOT,
        HDF5_DATA_FILE,
        "--subset-names", 
        "MineRLTreechopVectorObf-v0",
        "MineRLObtainIronPickaxeVectorObf-v0"
    ]
    #store_subset_to_hdf5(store_subset_to_hdf5_params)

    aicrowd_helper.register_progress(0.20)

    # Fit Kmeans on actions
    # Suuuuuper-elegant argument passing, thanks
    # to the big-brain use of argparse
    kmean_params = [
        HDF5_DATA_FILE,
        ACTION_CENTROIDS_FILE,
        "--n-clusters", "150",
        "--n-init", "30"
    ]
    fit_kmeans(kmean_params)

    aicrowd_helper.register_progress(0.40)

    # Update centroid locations in the data
    update_hdf5_params = [
        HDF5_DATA_FILE,
        ACTION_CENTROIDS_FILE
    ]
    update_hdf5_with_centroids(update_hdf5_params)

    aicrowd_helper.register_progress(0.60)

    # Remove frameskipped frames for LSTM training
    removed_frameskipped_params = [
        HDF5_DATA_FILE,
        HDF5_DATA_FILE_FRAMESKIPPED
    ]
    remove_frameskipped_samples(removed_frameskipped_params)

    aicrowd_helper.register_progress(0.80)

    # Train model with behavioural cloning
    bc_train_params = [
        HDF5_DATA_FILE_FRAMESKIPPED,
        TRAINED_MODEL_PATH,
        "--num-epochs", "1",
        "--include-frameskip", "16",
        "--discrete-actions",
        "--num-discrete-actions", "150",
        "--frameskip-from-vector",
        "--batch-size", "32",
        "--lr", "0.0000625",
        "--weight-decay", "1e-5",
        "--seq-len", "32",
        "--horizontal-flipping",
        "--entropy-weight", "0.0",
        "--resnet", "ResNetHeadFor64x64DoubleFilters"
    ]
    parsed_args, remaining_args = train_bc_parser.parse_known_args(bc_train_params)
    main_train_bc(parsed_args, remaining_args)
    aicrowd_helper.register_progress(1.0)
    aicrowd_helper.training_end()

if __name__ == "__main__":
    main()
