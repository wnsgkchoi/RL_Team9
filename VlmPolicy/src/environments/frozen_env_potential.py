import json
import pathlib
from enum import Enum
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium import Wrapper
from gymnasium.envs.toy_text.frozen_lake import (FrozenLakeEnv,
                                                 generate_random_map)
from gymnasium.wrappers import RecordVideo
from torchvision.transforms.functional import resize

from environments.utils import VisualObsWrapper


class FrozenActions(Enum):
    move_west = "move west"
    move_south = "move south"
    move_east = "move east"
    move_north = "move north"


class FrozenLakeTextPotential(FrozenLakeEnv):    
    """Wrapper for the FrozenLake environment that returns text observations and uses Potential-based Reward Shaping."""
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}
    def __init__(self, map_size=8, fov=3, fixed_orientation=False, is_slippery=True, seed=0, first_person=False, gamma=0.99):
        desc = generate_random_map(size=map_size, seed=seed)
        super().__init__(desc=desc, is_slippery=is_slippery, render_mode='rgb_array')
        self.seed = seed
        self.fov = fov
        self.fixed_orientation = fixed_orientation
        self.is_slippery = is_slippery
        self.first_person = first_person
        self.gamma = gamma # Discount factor for potential shaping
        self.size = map_size

        self.action_enum = FrozenActions

        self.action2label = {0: 'move west', 1: 'move south', 2: 'move east', 3: 'move north'}
        self.action2alpha = {0: np.pi, 1: -np.pi/2, 2: 0, 3: np.pi/2}
        
        self.holes = []
        for row, col in [(i,j) for i in range(self.nrow) for j in range(self.ncol)]:
            if self.desc[row, col] == b'H':
                self.holes.append(self._get_pos(row*self.ncol + col))
            if self.desc[row, col] == b'G':
                self.goal = self._get_pos(row*self.ncol + col)
        
        self.walls = [np.array([-1, y]) for y in range(-1, self.nrow+1)]
        self.walls += [np.array([self.ncol, y]) for y in range(-1, self.nrow+1)]
        self.walls += [np.array([x, -1]) for x in range(self.ncol)]
        self.walls += [np.array([x, self.nrow]) for x in range(self.ncol)]        
        
        # Pre-calculate BFS distances for Potential
        self._compute_bfs_potential()

    def _compute_bfs_potential(self):
        """Compute shortest path distance from every cell to the goal using BFS."""
        self.bfs_distance_map = np.full((self.nrow, self.ncol), fill_value=self.nrow * self.ncol, dtype=np.float32)
        
        # Find goal coordinates (row, col)
        goal_indices = np.where(self.desc == b'G')
        if len(goal_indices[0]) == 0:
            return # Should not happen
        goal_r, goal_c = goal_indices[0][0], goal_indices[1][0]
        
        queue = deque([(goal_r, goal_c, 0)])
        visited = set([(goal_r, goal_c)])
        self.bfs_distance_map[goal_r, goal_c] = 0
        
        while queue:
            r, c, dist = queue.popleft()
            
            # Check 4 directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.nrow and 0 <= nc < self.ncol:
                    if (nr, nc) not in visited and self.desc[nr, nc] != b'H':
                        visited.add((nr, nc))
                        self.bfs_distance_map[nr, nc] = dist + 1
                        queue.append((nr, nc, dist + 1))

    def _get_pos(self, s):
        return np.array([s % self.ncol, self.nrow - (s // self.ncol) - 1])
    
    def _get_potential(self, s):
        """Calculate potential phi(s) = -BFS_Distance(s, goal)"""
        # Convert state s to (row, col)
        row = s // self.ncol
        col = s % self.ncol
        
        dist = self.bfs_distance_map[row, col]
        
        # Normalize to [0, 1] range roughly, using max possible path
        # A safe upper bound for normalization is size * size
        norm_dist = dist / (self.nrow * self.ncol)
        
        return -norm_dist

    def _get_entity_desc(self, entity_pos, title):
        pos = self._get_pos(self.s)
        entity_alpha = np.angle((entity_pos - pos) @ np.array([1, 1j]))
        alpha = (np.pi/2 if self.fixed_orientation else  # if fixed_orientation==True, the orientation is always north
                 self.action2alpha.get(self.last_action, -np.pi/2))
        delta_alpha = entity_alpha - alpha
        if delta_alpha >= 2* np.pi:
            delta_alpha -= 2*np.pi
        if delta_alpha < 0:
            delta_alpha += 2*np.pi
        
        if delta_alpha == 0 or delta_alpha == 2*np.pi:
            direction = 'north'
        elif delta_alpha < np.pi/2:
            direction = 'north-west'
        elif delta_alpha == np.pi/2:
            direction = 'west'
        elif delta_alpha < np.pi:
            direction = 'south-west'
        elif delta_alpha == np.pi:
            direction = 'south'
        elif delta_alpha < 3*np.pi/2:
            direction = 'south-east'
        elif delta_alpha == 3*np.pi/2:
            direction = 'east'
        elif delta_alpha < 2*np.pi:
            direction = 'north-east'
        
        steps = np.abs(entity_pos - pos).sum()
        if steps > self.fov and title != "the goal":
            return ''
        
        if steps == 0:
            res = f"- {title} right under {'me' if self.first_person else 'you'}.\n"
        else:
            res = f"- {title} {steps} steps to {'my' if self.first_person else 'your'} {direction}.\n"
        return res
    
    def _get_text_obs(self):
        res = f"{'I' if self.first_person else 'You'} took action {self.action2label.get(self.last_action, 'noop')}"
        if self.is_slippery and self.last_effective_action is not None and self.last_effective_action != self.last_action:
            res += f" but {'I' if self.first_person else 'you'} slipped, and {'I' if self.first_person else 'you'} took action {self.action2label[self.last_effective_action]}"
        
        res += f'.\n{"I" if self.first_person else "You"} see:\n'
        for wall_pos in self.walls:
            res += self._get_entity_desc(wall_pos, "a wall")
        for hole_pos in self.holes:
            res += self._get_entity_desc(hole_pos, "a trap")
        
        res += self._get_entity_desc(self.goal, "the goal")
        return res
    
    def _get_history(self):
        history = f'{len(self.history[:2])} most recent observations:\n\n'
        for i, h in enumerate(self.history[:2]):
            history += f"{'My' if self.first_person else 'Your'} observation at step {i}:\n{h}\n"
        return history[:-1]
    
    def step(self, action):
        action = int(action)
        self.prev_s = self.s
        
        obs, reward, done, truncated, info = super().step(action)
        
        # Base Reward (R_env)
        if done and reward == 0:  # Fell into a hole (Trap)
            r_env = -1.0
            info['is_success'] = False
        elif done and reward == 1:  # Reached the goal
            r_env = 1.0
            info['is_success'] = True
        else:
            r_env = -0.01 # Small step penalty to encourage shorter paths
            info['is_success'] = False
            
        # Potential-based Reward Shaping
        # F = gamma * phi(s') - phi(s)
        phi_t = self._get_potential(self.prev_s)
        phi_t1 = self._get_potential(self.s)
        
        shaping = self.gamma * phi_t1 - phi_t
        
        # Total Reward
        reward = r_env + shaping
            
        self.last_action = action
        
        # get the effective action
        x, y = self._get_pos(self.s) - self._get_pos(self.prev_s)
        if x > 0:
            self.last_effective_action = 2  # right
        elif x < 0:
            self.last_effective_action = 0  # left 
        elif y > 0:
            self.last_effective_action = 3  # up
        elif y < 0:
            self.last_effective_action = 1  # down
        
        # update info
        info['obs'] = self._get_text_obs()
        self.history.append(info['obs'])
        info['history'] = self._get_history()
        
        # Add shaping info for debugging
        info['shaping_reward'] = shaping
        info['base_reward'] = r_env
        
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, **kwargs):
        obs, info = super().reset(seed=self.seed if seed is None else seed)
        self.last_action = None
        self.last_effective_action = None
        self.prev_s = None
        
        # update info
        info['obs'] = self._get_text_obs()
        self.history = [info['obs']]
        info['history'] = self._get_history()
        
        return obs, info


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
            is_success = info.get('is_success', False)
            self._stats = {'length': self._length, 'reward': round(self._reward, 1), 'is_success': is_success}
            self._save()
        return obs, reward, terminated, truncated, info

    def _save(self):
        self._file.write(json.dumps(self._stats) + '\n')
        self._file.flush()


def make_frozen_env_potential(
    outdir,
    area=8, # 8x8
    fov=1,
    is_slippery=False,
    fixed_orientation=True,
    save_video=False,
    size=(244, 244),
    save_video_every=100,
    seed=None,
    save_stats=False,
    first_person=True,
    gamma=0.99
):   
    def resize_obs(obs):
        obs = torch.from_numpy(obs).permute(2, 0, 1)
        obs = resize(obs, size, antialias=False)
        obs = obs.permute(1, 2, 0).numpy()
        return obs
    
    def frozen_thunk():
        frozen_env = gym.make('FrozenLakeText-Potential-v0', map_size=area, is_slippery=is_slippery, seed=seed, fov=fov, fixed_orientation=fixed_orientation, max_episode_steps=100, first_person=first_person, gamma=gamma)
        frozen_env = VisualObsWrapper(frozen_env, transform=resize_obs)
        if save_video:
            frozen_env = RecordVideo(frozen_env, video_folder=outdir, episode_trigger=lambda ix: ix % save_video_every == 0)
        if save_stats:
            frozen_env = StatsRecorder(frozen_env, stat_folder=outdir)
        return frozen_env
    
    return frozen_thunk
