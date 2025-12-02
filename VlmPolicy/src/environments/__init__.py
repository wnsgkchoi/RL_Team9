import gymnasium
from .frozen_env import FrozenLakeText
from .hanoi_env import Hanoi3Disk, Hanoi4Disk

gymnasium.register('FrozenLakeText-v0', entry_point=FrozenLakeText, kwargs={'map_size': 8, 'is_slippery': False, 'seed': 0, 'fov': 1, 'fixed_orientation': True})
gymnasium.register('Hanoi3Disk-v0', entry_point=Hanoi3Disk, max_episode_steps=100)
gymnasium.register('Hanoi4Disk-v0', entry_point=Hanoi4Disk, max_episode_steps=100)