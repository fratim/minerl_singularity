#
# train_bc.py
#
# Train simple behavioural cloning on the dataset
#

from argparse import ArgumentParser
from collections import deque
from functools import partial
from time import time

import numpy as np
import torch
from tqdm import tqdm
from gym import spaces

from dataloaders.hdf5_loader import HDF5Loader, HDF5RandomSampler
from torch_codes.modules import IMPALANetwork
from utils.misc_utils import parse_keyword_arguments
from wrappers.action_wrappers import CentroidActions
from wrappers.observation_wrappers import resize_image

RESNETS = [
    "ResNetHeadFor64x64",
    "ResNetHeadFor32x32",
    "ResNetHeadFor64x64DoubleFilters",
    "ResNetHeadFor64x64QuadrupleFilters",
    "ResNetHeadFor64x64DoubleFiltersWithExtra"
]

parser = ArgumentParser("Train PyTorch networks on MineRL data with behavioural cloning.")
parser.add_argument("hdf5_file", type=str, help="MineRL dataset as a HDF5 file (see utils/handle_dataset.py)")
parser.add_argument("output", type=str, help="Where to store the trained model")
parser.add_argument("--batch-size", type=int, default=64, help="Yer standard batch size")
parser.add_argument("--num-epochs", type=int, default=5, help="Number of times to go over the dataset")
parser.add_argument("--include-frameskip", type=int, default=None, help="If provided, predict frameskip and this is max")
parser.add_argument("--lr", type=float, default=0.0005, help="Good old learning rate for Adam")
parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for Adam (i.e. L2 loss)")
parser.add_argument("--image-size", type=int, default=64, help="Resized image shape (original is 64)")
parser.add_argument("--resnet", type=str, default="ResNetHeadFor64x64", choices=RESNETS, help="ResNet type to use for images")

parser.add_argument("--discrete-actions", action="store_true", help="Use discrete actions (inside hdf5 file) rather than normal latents")
parser.add_argument("--frameskip-from-vector", action="store_true", help="Use frameskip targets based on action vectors, not discrezited actions")
parser.add_argument("--num-discrete-actions", type=int, default=100, help="DIRTY way of providing number of discrete options, for now")


def main(args, unparsed_args):
    # Create dataloaders
    resize_func = None if args.image_size == 64 else partial(resize_image, width_and_height=(args.image_size, args.image_size))
    dataset_mappings = {
        "observations/pov": ("pov", resize_func),
        "observations/vector": ("obs_vector", None),
    }
    if args.discrete_actions:
        dataset_mappings["actions/discrete_actions"] = ("action", None)
        if args.frameskip_from_vector:
            dataset_mappings["actions/num_action_repeated"] = ("frameskip", None)
        else:
            dataset_mappings["actions/num_action_repeated_discrete"] = ("frameskip", None)
    else:
        # Standard action vectors
        dataset_mappings["actions/vector"] = ("action", None)
        dataset_mappings["actions/num_action_repeated"] = ("frameskip", None)

    # If None, use all samples. Otherwise
    # load this dataset and use as boolean mask
    # to select valid samples
    valid_indeces_mask = None
    if args.include_frameskip:
        if args.discrete_actions and not args.frameskip_from_vector:
            valid_indeces_mask = "actions/start_of_new_action_discrete"
        else:
            valid_indeces_mask = "actions/start_of_new_action"

    data_sampler = HDF5RandomSampler(
        args.hdf5_file,
        dataset_mappings,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        valid_indeces_mask=valid_indeces_mask,
        num_workers=1
    )

    # Create temporary HDF5Loader to get the types
    temp_loader = HDF5Loader(args.hdf5_file, dataset_mappings)
    # TODO read n_discrete_actions here
    shapes_and_types = temp_loader.get_types()
    temp_loader.close()

    # This is hardcoded in PyTorch format
    image_shape = (3, args.image_size, args.image_size)

    num_additional_features = shapes_and_types["obs_vector"]["shape"][0]

    # Define the action_space so we know to do scaling etc later,
    # as well as how many scalars we need from network
    action_space = None
    num_action_outputs = None
    if args.discrete_actions:
        # TODO need prettier way to tell what is the maximum action
        num_action_outputs = args.num_discrete_actions
        action_space = spaces.Discrete(n=num_action_outputs)
    else:
        # Standard latents
        # Default number of outputs for regressing directly on
        # action vectors
        num_action_outputs = shapes_and_types["action"]["shape"][0]
        action_space = spaces.Box(shape=(num_action_outputs,), dtype=np.float32, low=1.05, high=1.05)

    output_dict = {
        "action": num_action_outputs
    }
    if args.include_frameskip is not None:
        # Tell that we need also one-hot output
        output_dict["frameskip"] = args.include_frameskip

    # Bit of sanity checking
    if args.resnet != "ResNetHeadFor32x32" and args.image_size < 64:
        raise ValueError("Using a big network for smaller image. You suuuuure you want to do that?")

    network = IMPALANetwork(image_shape, output_dict, num_additional_features, cnn_head_class=args.resnet).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    losses = deque(maxlen=1000)
    start_time = time()

    for i, data_batch in enumerate(tqdm(data_sampler, desc="train")):
        # Do transposing of the images
        network_output = network(
            torch.from_numpy(data_batch["pov"].transpose(0, 3, 1, 2)).cuda(),
            torch.from_numpy(data_batch["obs_vector"]).cuda(),
        )

        total_loss = 0

        # pi-loss (i.e. predict correct action)
        predicted_action = network_output["action"]
        target_action = data_batch["action"]
        if args.discrete_actions:
            # Fix there somewhere more sensible?
            target_action = target_action.astype(np.int64).reshape((-1))
        target_action = torch.from_numpy(target_action).cuda()

        if isinstance(action_space, spaces.Box):
            # Squeeze actions to [0, 1] and scale to
            # whatever the target actions have
            predicted_action = torch.sigmoid(predicted_action)
            predicted_action = (predicted_action * (action_space.high - action_space.low)) + action_space.low
            # Simple mean absolute-error (or L1 norm)
            total_loss += torch.mean(torch.abs(target_action - predicted_action))
        elif isinstance(action_space, spaces.Discrete):
            # Standard cross-entropy
            total_loss += torch.nn.functional.cross_entropy(predicted_action, target_action)
        else:
            raise NotImplementedError

        if args.include_frameskip is not None:
            # Action-frameskip loss
            predicted_frameskip = network_output["frameskip"]
            target_frameskip = data_batch["frameskip"]
            target_frameskip = torch.from_numpy(target_frameskip.ravel()).long().cuda()
            # Move frameskip=1 to be in index 1, also clip at higher end
            target_frameskip = torch.clamp(target_frameskip, 1, args.include_frameskip) - 1
            # TODO add label smoothing etc here?
            total_loss += torch.nn.functional.cross_entropy(predicted_frameskip, target_frameskip)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.cpu().detach().item())

        if (i % 10000) == 0:
            tqdm.write("Steps {:<9} Time {:<9} Avrg loss {:<10.5f}".format(
                i,
                int(time() - start_time),
                sum(losses) / len(losses)
            ))

            # TODO consider using state_dict variant,
            #      to avoid any pickling issues etc
            torch.save(network, args.output)

    torch.save(network, args.output)


if __name__ == "__main__":
    args, unparsed_args = parser.parse_known_args()
    main(args, unparsed_args)
