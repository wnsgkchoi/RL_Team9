import gymnasium
import torch
import copy
import json
import pathlib
from gymnasium import Wrapper
from torchvision.transforms.functional import resize


class VisualObsWrapper(gymnasium.Wrapper):
    """Wrapper that returns visual observations from the environment."""
    def __init__(self, env, transform=None):
        super(VisualObsWrapper, self).__init__(env)
        self.transform = transform
        dummy_obs, _ = self.reset()
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

    def _render_transform(self):
        obs = self.env.render()
        if self.transform is not None:
            obs = self.transform(obs)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._render_transform()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self._render_transform()
        return obs, reward, done, truncated, info
    
class HistoryTracker:

    def __init__(self, max_steps) -> None:
        self.max_steps = max_steps
        self.game_step = 0
        self.reset()

    def step(self, info) -> None:
        self.info.append(copy.copy(info))
        if len(self.info) > self.max_steps:
            self.info.pop(0)
        self.game_step += 1

    def reset(self) -> None:
        self.info = []
        self.game_step = 0
    
    def describe(self, game_step=None):
        if len(self.info) == 0:
            return ""
        game_step = self.game_step if game_step is None else game_step
        result = "Most recent {} steps of the player's in-game observation:\n\n".format(len(self.info))
        for i, info in enumerate(self.info):
            result += "Player Observation Step {}:\n".format(game_step - len(self.info) + i)
            result += info["obs"] + "\n\n"
        return result.strip()
    
    def score(self):
        return sum([info["score"] for info in self.info])

class StatsRecorder(Wrapper):
    def __init__(self, env, stat_folder):
        super().__init__(env)
        self._directory = pathlib.Path(stat_folder).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        self._file = (self._directory / 'stats.jsonl').open('a')
        self._length = None
        self._reward = None
        self._stats = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._length = 0
        self._reward = 0
        self._stats = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._length += 1
        self._reward += reward
        done = terminated or truncated
        if done:
            self._stats = {'length': self._length, 'reward': round(self._reward, 1)}
            self._save()
        return obs, reward, terminated, truncated, info

    def _save(self):
        self._file.write(json.dumps(self._stats) + '\n')
        self._file.flush()

def describe_act(action_list):
    return "List of all actions:\n" + "\n".join(["{}. {}".format(i+1, s) for i,s in enumerate(action_list)])
