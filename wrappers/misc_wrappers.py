from copy import deepcopy
import time
import datetime
import warnings
import os
import orjson

import gym
from gym import Wrapper

from utils.monitoring import VideoRecorder


class SuperviserWrapper(Wrapper):
    """
    Wrapper that reboots the environment if it happens to crash.

    If environment crashes during step, we return the latest
    observation as a terminal state, and re-init the environment.

    Note: Env should be creatable with gym.make
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.env_id = env.unwrapped.spec.id

        # Keep track of last observation so we have something
        # valid to give to the agent
        self.last_obs = None

        self.start_time = time.time()

    def reboot_env(self):
        """Re-create the environment from zero"""
        # See if we can close the environment
        try:
            self.env.close()
        except Exception:
            pass
        self.env = gym.make(self.env_id)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs

    def step(self, action):
        obs, reward, done, info = None, None, None, None

        try:
            obs, reward, done, info = self.env.step(action)
            self.last_obs = obs
        except Exception as e:
            print("[{}, Superviser] Environment crashed with '{}'".format(
                time.time() - self.start_time, str(e))
            )
            # Create something to return
            obs = self.last_obs
            reward = 0
            done = True
            info = {}
            # Re-create the environment
            self.reboot_env()

        return obs, reward, done, info


class RecordWrapper(Wrapper):
    """
    Wrapper for recording everything of MineRL
    games.

    Information about steps and all is stored
    in `[recording_dir]/[recording_name]_[timestamp]/`
    """

    # Video name without postfix
    VIDEO_NAME = "video"
    OBSERVATIONS_JSON_NAME = "observations.json"
    ACTIONS_JSON_NAME = "actions.json"
    REWARDS_JSON_NAME = "rewards.json"
    MISC_JSON_NAME = "misc.json"

    def __init__(self, env, recording_dir, recording_name):
        if isinstance(env, Wrapper):
            warnings.warn("RecordWrapper should be first wrapper for best recordings.")
        super().__init__(env)
        self.env = env
        self.recording_dir = recording_dir
        self.recording_name = recording_name
        self.current_recording_dir = None
        self.current_video_recorder = None
        self.env_step = 0

        # This is for storing inventories and currently
        # equipped item
        self.recorded_observations = []
        self.recorded_actions = []
        self.recorded_rewards = []
        self.recorded_misc = []

    def _save_recordings(self):
        """Save all current data to disk"""
        self.current_video_recorder.finish_video()
        # Save other stuff as jsons
        # Yummy duping
        json_path = os.path.join(
            self.current_recording_dir,
            RecordWrapper.OBSERVATIONS_JSON_NAME
        )
        with open(json_path, "wb") as f:
            f.write(orjson.dumps(self.recorded_observations, option=orjson.OPT_SERIALIZE_NUMPY))
        json_path = os.path.join(
            self.current_recording_dir,
            RecordWrapper.ACTIONS_JSON_NAME
        )
        with open(json_path, "wb") as f:
            f.write(orjson.dumps(self.recorded_actions, option=orjson.OPT_SERIALIZE_NUMPY))
        json_path = os.path.join(
            self.current_recording_dir,
            RecordWrapper.REWARDS_JSON_NAME
        )
        with open(json_path, "wb") as f:
            f.write(orjson.dumps(self.recorded_rewards, option=orjson.OPT_SERIALIZE_NUMPY))
        json_path = os.path.join(
            self.current_recording_dir,
            RecordWrapper.MISC_JSON_NAME
        )
        with open(json_path, "wb") as f:
            f.write(orjson.dumps(self.recorded_misc, option=orjson.OPT_SERIALIZE_NUMPY))

    def reset(self, **kwargs):
        self.env_step = 0
        if self.current_video_recorder is not None:
            self._save_recordings()

        # Create new recording
        self.current_recording_dir = os.path.join(
            self.recording_dir,
            # Windows does not like ":" in paths
            "{}_{}".format(
                self.recording_name,
                datetime.datetime.now().isoformat().replace(":", "-")
            ),
        )
        os.makedirs(self.current_recording_dir)

        obs = self.env.reset(**kwargs)

        self.current_video_recorder = VideoRecorder(
            width=obs["pov"].shape[0],
            height=obs["pov"].shape[1],
            save_path=os.path.join(self.current_recording_dir, RecordWrapper.VIDEO_NAME)
        )
        # Record first frame
        self.current_video_recorder.save_frame(obs["pov"])

        self.recorded_actions.clear()
        self.recorded_rewards.clear()
        self.recorded_observations.clear()

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.env_step += 1

        self.current_video_recorder.save_frame(obs["pov"])
        # Record non-pov data
        self.recorded_observations.append(
            dict([[key, value] for key, value in obs.items() if key != "pov"])
        )
        self.recorded_actions.append(action)
        self.recorded_rewards.append(reward)

        return obs, reward, done, info

    def record_misc(self, misc_dict):
        """
        Store data into misc json file.

        Parameters:
            misc_dict (dict): A dictionary of info to be stored
        """
        # Include step
        misc_dict["_env_step"] = self.env_step
        self.recorded_misc.append(misc_dict)

    def close(self):
        self._save_recordings()
