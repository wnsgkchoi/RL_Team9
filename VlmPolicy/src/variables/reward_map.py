import json
import os
import numpy as np


class RewardMaps:
    def __init__(self):
        self.reward_maps = {}

    def set_reward_map(self, env_number, reward_map):
        self.reward_maps[env_number] = reward_map

    def get_reward_map(self, env_number):
        return self.reward_maps[env_number]
    
    def save_to_file(self, save_dir, process_index=0):
        """
        Save reward maps to a JSON file.
        
        Args:
            save_dir: Directory to save the reward maps
            process_index: Process index for multi-process training
        """
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'reward_maps_process_{process_index}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_maps = {}
        for env_idx, reward_map in self.reward_maps.items():
            if isinstance(reward_map, np.ndarray):
                serializable_maps[str(env_idx)] = reward_map.tolist()
            elif isinstance(reward_map, list):
                serializable_maps[str(env_idx)] = reward_map
            else:
                serializable_maps[str(env_idx)] = reward_map
        
        with open(filepath, 'w') as f:
            json.dump(serializable_maps, f, indent=2)
        
        print(f"[RewardMaps] Saved {len(self.reward_maps)} reward maps to {filepath}")
    
    def load_from_file(self, save_dir, process_index=0):
        """
        Load reward maps from a JSON file.
        
        Args:
            save_dir: Directory where reward maps are saved
            process_index: Process index for multi-process training
            
        Returns:
            bool: True if file was found and loaded, False otherwise
        """
        filepath = os.path.join(save_dir, f'reward_maps_process_{process_index}.json')
        
        if not os.path.exists(filepath):
            print(f"[RewardMaps] No saved reward maps found at {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                loaded_maps = json.load(f)
            
            # Convert lists back to numpy arrays and int keys
            self.reward_maps = {}
            for env_idx_str, reward_map in loaded_maps.items():
                env_idx = int(env_idx_str)
                self.reward_maps[env_idx] = reward_map  # Keep as list or convert to numpy as needed
            
            print(f"[RewardMaps] Loaded {len(self.reward_maps)} reward maps from {filepath}")
            return True
        except Exception as e:
            print(f"[RewardMaps] Error loading reward maps: {e}")
            return False
    
    def has_reward_map(self, env_number):
        """
        Check if a reward map exists for the given environment number.
        
        Args:
            env_number: Environment index
            
        Returns:
            bool: True if reward map exists, False otherwise
        """
        return env_number in self.reward_maps



reward_maps = RewardMaps()
