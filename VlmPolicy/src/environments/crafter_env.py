import gymnasium as gym
import pathlib
import imageio

from smartplay.crafter.crafter_env import Crafter
from crafter.recorder import StatsRecorder, EpisodeRecorder, EpisodeName
from enum import Enum


class CrafterActions(Enum):
    noop = "don't do anything"
    move_left = "move one step west"
    move_right = "move one step east"
    move_up = "move one step north"
    move_down = "move one step south"
    do = "interact with what is in front of you"
    sleep = "just take a nap"
    place_stone = "place a stone in front of you"
    place_table = "place a table in front of you"
    place_furnace = "place a furnace in front of you"
    place_plant = "place a plant in front of you"
    make_wood_pickaxe = "make a woody pickaxe"
    make_stone_pickaxe = "make a stone pickaxe"
    make_iron_pickaxe = "make an iron pickaxe"
    make_wood_sword = "make a woody sword"
    make_stone_sword = "make a stone sword"
    make_iron_sword = "make an iron sword"

class VideoRecorder:

    def __init__(self, env, directory, size=(512, 512), save_video_every=100):
        if not hasattr(env, "episode_name"):
            env = EpisodeName(env)
        self._env = env
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        self._size = size
        self._frames = None
        self._episode_id = 0
        self._save_video_every = save_video_every
        print(f"recording video in {self._directory}")

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        self._episode_id += 1
        if self._episode_id % self._save_video_every == 1:
            self._frames = [self._env.render(self._size)]
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        if self._episode_id % self._save_video_every == 1:
            self._frames.append(self._env.render(self._size))
            if done:
                self._save()
        return obs, reward, done, info

    def _save(self):
        filename = str(self._directory / (self._env.episode_name + ".mp4"))
        imageio.mimsave(filename, self._frames)


class Custom_Recorder:
    """
    crafter Recorder
    """

    def __init__(
        self,
        env,
        directory,
        save_stats=True,
        save_video=True,
        save_episode=True,
        video_size=(512, 512),
        save_video_every=100
    ):
        if directory and save_stats:
            env = StatsRecorder(env, directory)
        if directory and save_video:
            env = VideoRecorder(env, directory, video_size, save_video_every)
        if directory and save_episode:
            env = EpisodeRecorder(env, directory)
        self._env = env

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._env, name)


def make_crafter_env(
    outdir,
    save_stats=False,
    save_episode=False,
    save_video=False,
    save_video_every=100,
    area=(64, 64),
    view=(9, 9),
    size=(64, 64),
    reward=True, 
    length=10000, 
    seed=None, 
    max_steps=2
):
    def crafter_thunk():
        crafter_env = Crafter(area=area, view=view, size=size, reward=reward, length=length, seed=seed, max_steps=max_steps) 
        crafter_env = Custom_Recorder(
            crafter_env,
            outdir,
            save_stats=save_stats,
            save_episode=save_episode,
            save_video=save_video,
            save_video_every=save_video_every
        )
        env = gym.make("GymV21Environment-v0", env=crafter_env)
        env.action_enum = CrafterActions

        return env

    return crafter_thunk

