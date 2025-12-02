import json
import pathlib
import gymnasium
import numpy as np
import torch
import minigrid

from enum import Enum, IntEnum
from gymnasium import Wrapper
from gymnasium.wrappers import RecordVideo
from gymnasium.core import ActionWrapper
from gymnasium.spaces import Discrete
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, ReseedWrapper
from minigrid.core.actions import Actions

from .utils import StatsRecorder

class TextualActions(Enum):
    # Turn left, turn right, move forward
    left = "turn left" #0
    right = "turn right" #1
    forward = "move forward" #2
    # Pick up an object
    pickup = "pick up" #3
    # Drop an object
    drop = "drop" #4
    # Toggle/activate an object
    toggle = "toggle" #5
    # Done completing task
    done = "done" #6

    # OPTIONS
    opt_left = "move left" # 7
    opt_right = "move right" # 8
    opt_back = "move back" # 9


class EnvTranslator:
    """
    Class used to generate textual description of minigrid envs.
    """
    @staticmethod
    def assert_valid_env(env):
        if not isinstance(env.unwrapped, MiniGridEnv):
            raise ValueError("The environment must be an instance of MiniGridEnv or its subclass.")

    @staticmethod
    def get_tile_location(env):#, tile_type):
        EnvTranslator.assert_valid_env(env)
        finded = []
        for j in range(env.unwrapped.grid.height):
            for i in range(env.unwrapped.grid.width):
                tile = env.unwrapped.grid.get(i, j)
                if tile: #and tile.type == tile_type:
                    finded.append(np.array([i, j]))
        return finded
    
    @staticmethod
    def get_tile_location_description(env, agent_position, all_tile_position, fov=1, fixed_orientation=True, no_step_description=False, first_person=False):
        all_location_descriptions = []
        for tile_position in all_tile_position:
            tile_type = env.unwrapped.grid.get(*tile_position).type
            tile_direction = EnvTranslator.get_entity_direction(fixed_orientation, np.array([agent_position[0], env.unwrapped.grid.height-agent_position[1]]), env.unwrapped.agent_dir, np.array([tile_position[0], env.unwrapped.grid.width-tile_position[1]]), first_person)
            steps_to_tile = EnvTranslator.get_manhattan_distance(np.array([agent_position[0], env.unwrapped.grid.height-agent_position[1]]), np.array([tile_position[0], env.unwrapped.grid.width-tile_position[1]]))
            if steps_to_tile == 0:
                tile_location_description = f"- {EnvTranslator.describe_tile(env.unwrapped.grid.get(*tile_position))} right under {'me' if first_person else 'you'}.\n"
            elif steps_to_tile == 1:
                step_descriptor = 'directly' if no_step_description else '1 step'
                tile_location_description = f"- {EnvTranslator.describe_tile(env.unwrapped.grid.get(*tile_position))} {step_descriptor} {tile_direction}.\n"
            elif steps_to_tile <= fov or tile_type in ['goal', 'key', 'door', 'box', 'ball']:
                step_descriptor = 'many' if no_step_description else steps_to_tile
                tile_location_description = f"- {EnvTranslator.describe_tile(env.unwrapped.grid.get(*tile_position))} {step_descriptor} steps {tile_direction}.\n"
            else:
                tile_location_description = ''
            all_location_descriptions.append(tile_location_description)
        return all_location_descriptions

    @staticmethod
    def get_manhattan_distance(agent_position, entity_pos):
        return np.abs(entity_pos - agent_position).sum()

    @staticmethod
    def get_entity_direction(fixed_orientation, agent_position, agent_direction, entity_pos, first_person):
        entity_alpha = np.angle((entity_pos - agent_position) @ np.array([1, 1j]))
        DIR_TO_ANGLE = {0: 0, 1: (3/2)*np.pi, 2: np.pi, 3: 0.5*np.pi}
        alpha = np.pi/2 if fixed_orientation else DIR_TO_ANGLE[agent_direction]
        delta_alpha = entity_alpha - alpha
        if delta_alpha >= 2* np.pi:
            delta_alpha -= 2*np.pi
        if delta_alpha < 0:
            delta_alpha += 2*np.pi
    
        if delta_alpha == 0 or delta_alpha == 2*np.pi:
            direction = 'to the north' if fixed_orientation else f'ahead of {"me" if first_person else "you"}'
        elif delta_alpha < np.pi/2:
            direction = 'to the north-west' if fixed_orientation else f'ahead of {"me" if first_person else "you"} to the left'
        elif delta_alpha == np.pi/2:
            direction = 'to the west' if fixed_orientation else f'to {"my" if first_person else "your"} left'
        elif delta_alpha < np.pi:
            direction = 'to the south-west' if fixed_orientation else f'behind {"me" if first_person else "you"}'
        elif delta_alpha == np.pi:
            direction = 'to the south' if fixed_orientation else f'behind {"me" if first_person else "you"}'
        elif delta_alpha < 3*np.pi/2:
            direction = 'to the south-east' if fixed_orientation else f'behind {"me" if first_person else "you"}'
        elif delta_alpha == 3*np.pi/2:
            direction = 'to the east' if fixed_orientation else f'to {"my" if first_person else "your"} right'
        elif delta_alpha < 2*np.pi:
            direction = 'to the north-east' if fixed_orientation else f'ahead of {"me" if first_person else "you"} to the right'
        return direction
    
    @staticmethod
    def describe_tile(tile):
        color = tile.color
        tile_description = ''
        OBJECT_TO_STR = {
                "wall": f"a {color} wall" if color else f"a wall",
                #"floor": "", we don't say anything for floor
                "key": f"a {color} key" if color else f"a key",
                "ball": f"a {color} ball" if color else f"a ball",
                "box": f"a {color} box" if color else f"a box",
                "goal": "the green goal square",
                "lava": f"{color} lava" if color else f"lava",
            }
        
        if tile.type == "door":
            if tile.is_open:
                tile_description = "an open door"
            elif tile.is_locked:
                tile_description = f"a locked {color} door"
            else:
                tile_description = f"a {color} door"
        else:
            tile_description = OBJECT_TO_STR[tile.type]
        
        return tile_description
    
    @staticmethod
    def describe_game_state_from_agent_perspective(env, fov=1, fixed_orientation=True, no_step_description=False, last_actions=[], first_person=False):
        EnvTranslator.assert_valid_env(env)
        if env.unwrapped.agent_pos is None or env.unwrapped.agent_dir is None or env.unwrapped.grid is None:
                raise ValueError(
                    "The environment hasn't been `reset` therefore the `agent_pos`, `agent_dir` or `grid` are unknown."
                )
        i = env.unwrapped.agent_pos[0] # column
        j = env.unwrapped.agent_pos[1] # row

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: "east", 1: "south", 2: "west", 3: "north"}

        #set the mission message
        state_description = f"{'My' if first_person else 'Your'} mission is:\n\n{env.unwrapped.mission}.\n\n"

        #surroundings
        state_description += f'{"I" if first_person else "You"} see:\n'

        all_locations = EnvTranslator.get_tile_location(env)
        all_location_descriptions = EnvTranslator.get_tile_location_description(env, (i,j), all_locations, fov=fov, fixed_orientation=fixed_orientation, no_step_description=no_step_description, first_person=first_person)

        for d in all_location_descriptions:
            state_description += d
        
        #direction and object carried 
        if fixed_orientation:
            state_description += f"\n{'I am' if first_person else 'You are'} facing {AGENT_DIR_TO_STR[env.unwrapped.agent_dir]} right now."
        if env.unwrapped.carrying is not None:
            tile_description = EnvTranslator.describe_tile(env.unwrapped.carrying)
            state_description += f" and {'I am' if first_person else 'you are'} carrying {tile_description}."
        else:
            state_description += f""
        
        if last_actions:
            state_description += f"\n\nLast actions {'I' if first_person else 'you'} took was: {', '.join(last_actions)}."

        return state_description

class TextObsWrapper(Wrapper):
    """
    Wrapper that uses EnvTranslator to return the state description 
    within info for both reset() and step().
    """
    def __init__(self, env, fov=1, fixed_orientation=True, no_step_description=False, history_size=0, first_person=False):
        super().__init__(env)
        self.fov = fov
        self.fixed_orientation = fixed_orientation
        self.first_person = first_person
        self.no_step_description = no_step_description
        self.history_size = history_size
        self.last_actions = []
        self.index_to_action = {i: action.value for i, action in enumerate(TextualActions)}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info['obs'] = EnvTranslator.describe_game_state_from_agent_perspective(self.env, fov=self.fov, fixed_orientation=self.fixed_orientation, no_step_description=self.no_step_description, first_person=self.first_person)
        self.last_actions = []
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action=action)
        # add action and keep only last history_size actions
        self.last_actions.append(action)
        self.last_actions = self.last_actions[-self.history_size:] if self.history_size else []  # list of actions (integers)
        info['obs'] = EnvTranslator.describe_game_state_from_agent_perspective(
            self.env,
            fov=self.fov,
            fixed_orientation=self.fixed_orientation,
            no_step_description=self.no_step_description,
            last_actions=[self.index_to_action[a] for a in self.last_actions],  # list of actions (strings),
            first_person=self.first_person
        )
        return obs, reward, terminated, truncated, info
    

class OptionWrapper(Wrapper):

    def __init__(self, env, possible_actions_list=None):
        super().__init__(env)

        # default to having all the original actions from minigrid env
        if possible_actions_list is None:
            possible_actions_list = list(TextualActions.__members__.keys())[:7]

        self.action_to_option_seq = {
            0: [0],
            1: [1],
            2: [2],
            3: [3],
            4: [4],
            5: [5],
            6: [6],
            7: [0, 2],  #opt_left -> left + forward
            8: [1, 2],  #opt_right -> right + forward
            9: [0, 0, 2],  #opt_back -> left + left + forward
         }

        # Check if all possible_actions are present among the original minigrid actions
        action_enum_names = set(member.name for member in TextualActions)
        assert all(action_name in action_enum_names for action_name in possible_actions_list)

        # Filter out the actions not needed from TextualActions
        self.action_enum = Enum('action_enum', {action.name: action.value for action in TextualActions if action.name in possible_actions_list})
        self.action_enum_indexes = [ix for ix, action in enumerate(TextualActions) if action.name in possible_actions_list]  # es: 0,1,4 derived from left,right,drop
        self.action_to_option_seq_filtered = self._filter_action_to_option()
    
    def _filter_action_to_option(self):
        action_to_option_seq_filtered = {}
        sorted_keys = sorted(self.action_enum_indexes)
        for i, key in enumerate(sorted_keys):
            action_to_option_seq_filtered[i] = self.action_to_option_seq[key]
        return action_to_option_seq_filtered
    
    def step(self, action):
        # get the action sequence from the requested option
        option_actions = self.action_to_option_seq_filtered[action] # 0 ->[0,2]

        # apply the actions
        for ix, act in enumerate(option_actions):
            # if we are doing an option we need to take care of the step count, since one option must account for 1 step
            if ix > 0:
                self.unwrapped.step_count -= 1
            obs, reward, terminated, truncated, info = self.env.step(action=act)
        return obs, reward, terminated, truncated, info


class RGBImgObsWrapper_fix(RGBImgObsWrapper):
    def __init__(self, env, tile_size=8):
        super().__init__(env)
        self.tile_size = tile_size
        new_image_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(self.env.height * tile_size, self.env.width * tile_size, 3),
            dtype="uint8",
        )
        self.observation_space = gymnasium.spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )


def make_minigrid_env(
    outdir,
    env_name, #'MiniGrid-LavaCrossingS11N5-v0'
    possible_actions_list=None,
    fov=1,
    fixed_orientation=True,
    no_step_description=False,
    seed=1,
    history_size=0,
    save_video=False,
    save_video_every=100,
    save_stats=False,
    reseed_env=False,
    first_person=False):   
    def minigrid_thunk():
        minigrid_env = gymnasium.make(env_name, render_mode="rgb_array")   # MiniGrid-LavaCrossingS11N5-v0, MiniGrid-DoorKey-8x8-v0
        minigrid_env = TextObsWrapper(minigrid_env, fov=fov, fixed_orientation=fixed_orientation, no_step_description=no_step_description, history_size=history_size, first_person=first_person)
        minigrid_env = RGBImgObsWrapper_fix(minigrid_env, tile_size=27)  #TODO tile_size=27 (default =8) just to increase the image dimension for VLMs. Maybe rescale with visualObsWrapper
        minigrid_env = ImgObsWrapper(minigrid_env) # obs is just the image, no mission or direction
        minigrid_env = ReseedWrapper(minigrid_env, seeds=[seed + s for s in range(10 if reseed_env else 1)]) # 10 different envs 
        minigrid_env = OptionWrapper(minigrid_env, possible_actions_list=possible_actions_list)
        if save_video:
            minigrid_env = RecordVideo(minigrid_env, video_folder=outdir, episode_trigger=lambda ix: ix % save_video_every==1)
        if save_stats:
            minigrid_env = StatsRecorder(minigrid_env, stat_folder=outdir)
        return minigrid_env
    return minigrid_thunk