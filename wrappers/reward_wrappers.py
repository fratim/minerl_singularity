from argparse import ArgumentParser

import numpy as np


def update_hdf5_with_reward_subtasks(remaining_args):
    """
    Take HDF5 file, and add (and replace) indeces
    for subtasks based on what is the next reward player will get.

    E.g. if next reward will be about obtaining crafting table, then
    the current subgoal is "craft crafting table".

    Adds datasets
        /subtask_ids (an integer specifying id of the next subtask)
    """
    import h5py
    from tqdm import tqdm

    parser = ArgumentParser("Deduce subtasks based on next reward")
    parser.add_argument("data", type=str, help="Path to the HDF5 file to be updated")
    args = parser.parse_args(remaining_args)

    data = h5py.File(args.data, "a")

    if "subtask_ids" in data:
        del data["subtask_ids"]
    rewards = data["rewards"][:][:, 0]
    dones = data["dones"][:]

    # Get all unique rewards and sort them to increasing order
    # (so we kind of know that higher id -> harder task)
    unique_rewards = sorted(list(set(rewards.tolist())))
    # Remove zero-reward
    del unique_rewards[unique_rewards.index(0)]

    subtask_ids = data.create_dataset("subtask_ids", shape=[rewards.shape[0], 1], dtype=np.uint8)
    subtask_ids_numpy = np.zeros(rewards.shape, dtype=np.uint8)

    # Go through rewards in reverse order, track the latest
    # reward and update subtask_ids to match whatever next
    # reward will be
    # We do not know subtask in the beginning, soooo lets set it high
    current_subtask = unique_rewards[-1]
    for i in tqdm(range(rewards.shape[0] - 1, -1, -1)):
        subtask_ids_numpy[i] = current_subtask

        reward = rewards[i]
        done = dones[i]
        if done:
            # Episode boundary. Set episode task to hardest possible
            current_subtask = unique_rewards[-1]
        if reward != 0:
            # This step gave reward and we have reached some
            # subgoal. Update that next steps will be for this
            # subgoal.
            # Include reward gained to the subtask
            current_subtask = unique_rewards.index(reward)

    # Copy data to h5py
    subtask_ids[:] = subtask_ids_numpy[:, None]

    data.close()


def print_num_subtasks(remaining_args):
    """
    Simply print out number of subtasks in the given HDF5 file.
    A lazy workaround for the fact that we are not reading this info
    from file when starting BC training...
    """
    import h5py

    parser = ArgumentParser("Print out number of subtasks in the HDF5 file")
    parser.add_argument("data", type=str, help="Path to the HDF5 file to be updated")
    args = parser.parse_args(remaining_args)

    data = h5py.File(args.data, "r")

    if "subtask_ids" not in data:
        print("No subtask_ids in the dataset")
    else:
        subtask_ids = data["subtask_ids"][:]
        # Assuming we go from {0..N}
        max_subtask_id = np.max(subtask_ids).item()
        print("Number of subtasks: {}".format(max_subtask_id + 1))


if __name__ == "__main__":
    VALID_OPERATIONS = {
        "update_subtasks": update_hdf5_with_reward_subtasks,
        "num_subtasks": print_num_subtasks,
    }
    parser = ArgumentParser("Run reward-wrapper related stuff")
    parser.add_argument("operation", type=str, choices=list(VALID_OPERATIONS.keys()), help="Operation to run")
    args, remainin_args = parser.parse_known_args()

    VALID_OPERATIONS[args.operation](remainin_args)
