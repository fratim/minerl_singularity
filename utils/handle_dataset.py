#
# Tools for modifying/handling the MineRL dataset (offline)
#

import numpy as np
from tqdm import tqdm
import minerl


def store_actions_to_numpy_file(subset_name, data_dir, output_file, num_workers=4):
    """
    Take all actions stored in the MineRL dataset under subset,
    gather them into big array and store in output_file

    Parameters:
        subset_name (str): Name of the dataset to process (e.g. MineRLTreechopVectorObf-v0)
        data_dir (str): Where MineRL dataset is stored.
        output_file (str): Where to store all the actions
        num_workers (int): Number of workers for data loader (default 4)
    """
    assert "Obf" in subset_name, "Must be obfuscated environment"

    data = minerl.data.make(subset_name, data_dir=data_dir, num_workers=num_workers)

    all_actions = []
    for _, actions, _, _, _ in tqdm(data.batch_iter(num_epochs=1, batch_size=32, seq_len=1)):
        all_actions.append(actions["vector"].squeeze())

    all_actions = np.concatenate(all_actions, axis=0)

    np.save(output_file, all_actions)


def store_subset_to_hdf5(subset_names, data_dir, output_file):
    """
    For "VectorObf" envs only!

    Convert all the samples from a datasets into a single HDF5
    file with different datasets for different variables:
        /observations
          /{any keys in the obs-space dict}
          ...
        /actions
          /vector
          /num_action_repeated (number of times action is repeated, 1 being smallest)
          /start_of_new_action (1 if this index is start of new action)
        /rewards                     (reward from executing action in last step)
        /dones
        /reward_to_go                (aka undiscounted return)
        /discounted_returns_.99      (returns discounted with gamma=0.99)
        /episodes/episode_starts     (where new episodes start)

    TODO There is no guarantee that the stored data will be sequential, or
         figure out a way to store data per episode.
         NOTE: Apparently this works like this

    Parameters:
        subset_names (List of str): Names of subsets to include in the data (e.g. MineRLTreechopVectorObf-v0)
        data_dir (str): Where MineRL dataset is stored.
        output_file (str): Where to store the HDF5 file
        num_workers (int): Number of workers for data loader (default 4)
    """
    assert all(map(lambda x: "Obf" in x, subset_names)), "Environments must be Obf envs"
    import gym
    import h5py

    datas = [minerl.data.make(subset_name, data_dir=data_dir, num_workers=1) for subset_name in subset_names]

    # First measure how many observations we have
    num_observations = 0
    for data in datas:
        for _, _, rewards, _, _ in tqdm(data.batch_iter(num_epochs=1, batch_size=1, seq_len=64), desc="size"):
            num_observations += rewards.shape[1]

    print("Total count of observations: {}".format(num_observations))

    # Assume that obs/action_spaces are dicts with only one depth
    obs_keys = list(data.observation_space.spaces.keys())
    obs_spaces = list(data.observation_space.spaces.values())
    act_space = data.action_space.spaces["vector"]

    # Create unified list of dataset names and their spaces
    dataset_keys = (
        list(map(lambda x: "observations/" + x, obs_keys)) +
        [
            "actions/vector",
            "actions/num_action_repeated",
            "actions/start_of_new_action",
            "rewards",
            "dones",
            "reward_to_go",
            "discounted_returns_.99"
        ]
    )
    dataset_spaces = (
        obs_spaces +
        [
            act_space,
            # Number of times action is repeated and where new actions begin
            gym.spaces.Box(shape=(1,), dtype=np.uint8, low=0, high=255),
            gym.spaces.Box(shape=(1,), dtype=np.uint8, low=0, high=1),
            # Reward and dones
            gym.spaces.Box(shape=(1,), dtype=np.float32, low=-np.inf, high=np.inf),
            gym.spaces.Box(shape=(1,), dtype=np.uint8, low=0, high=1),
            # Returns
            gym.spaces.Box(shape=(1,), dtype=np.float32, low=-np.inf, high=np.inf),
            gym.spaces.Box(shape=(1,), dtype=np.float32, low=-np.inf, high=np.inf),
        ]
    )
    datasets = {}

    # Create HDF5 file
    store_file = h5py.File(output_file, "w")
    # Create datasets
    for key, space in zip(dataset_keys, dataset_spaces):
        shape = (num_observations,) + space.shape
        dataset = store_file.create_dataset(key, shape=shape, dtype=space.dtype)
        datasets[key] = dataset

    # Read through dataset again and store items
    idx = 0
    last_action = None
    action_repeat_num = 0
    # Keep track where episodes start
    episode_starts = [0]

    for data in datas:
        for observations, actions, rewards, _, dones in tqdm(data.batch_iter(batch_size=1, num_epochs=1, seq_len=64), desc="store"):
            # Careful with the ordering of things here...
            # Iterate over seq len (second dim)
            for i in range(rewards.shape[1]):
                # Store different observations
                for key in obs_keys:
                    datasets["observations/{}".format(key)][idx] = observations[key][0, i]
                # Store action and measure how often they are repeated
                action = actions["vector"][0, i]
                datasets["actions/vector"][idx] = actions[key][0, i]
                # Check if action changed
                if last_action is None or not np.allclose(last_action, action):
                    if last_action is not None:
                        # Store how often the last action was repeated,
                        # each index telling how many times that action was
                        # going to be executed in future.
                        for j in range(1, action_repeat_num + 1):
                            datasets["actions/num_action_repeated"][idx - j] = j
                    last_action = action
                    action_repeat_num = 1
                else:
                    action_repeat_num += 1
                datasets["actions/start_of_new_action"][idx] = 1 if action_repeat_num == 1 else 0
                # Store other stuff
                datasets["rewards"][idx] = rewards[0, i]
                datasets["dones"][idx] = np.uint8(dones[0, i])
                if dones[0, i]:
                    last_action = None
                    episode_starts.append(idx + 1)

                idx += 1

    # Handle returns
    reward_to_go = 0
    current_return = 0
    dataset_reward_to_go = datasets["reward_to_go"]
    dataset_return = datasets["discounted_returns_.99"]
    rewards = datasets["rewards"]
    dones = datasets["dones"]
    for i in range(idx - 1, -1, -1):
        if dones[i]:
            reward_to_go = 0
            current_return = 0
        dataset_reward_to_go[i] = reward_to_go
        dataset_return[i] = current_return
        reward_to_go += rewards[i]
        current_return = current_return * 0.99 + rewards[i]

    # Add episode_starts dataset
    episode_starts_np = store_file.create_dataset("episodes/episode_starts", shape=(len(episode_starts),), dtype=np.int64)
    episode_starts_np[:] = np.array(episode_starts)[:]

    print("{} experiences moved into hdf5 file".format(idx))

    store_file.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("Convert MineRL dataset into HDF5 file")
    parser.add_argument("--subset-names", type=str, required=True, nargs="+", help="Name of the dataset to convert")
    parser.add_argument("data_dir", type=str, help="Location of MineRL dataset")
    parser.add_argument("output", type=str, help="Location where HDF5 file should be stored")
    args = parser.parse_args()
    store_subset_to_hdf5(args.subset_names, args.data_dir, args.output)
