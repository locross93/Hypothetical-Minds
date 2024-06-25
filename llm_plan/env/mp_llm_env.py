import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional

from meltingpot import substrate
import llm_plan.env.utils as utils


class MeltingPotLLMEnv:
    """A wrapper for melting pot environments to allow for LLM compatibility.
    Outputs a dictionary of states from an image. currently tested for global obs in
        - `commons_harvest__open`
    """
    def __init__(
            self, 
            substrate_name: str, 
            label_folder: Path,
            eval_num: int = None) -> None:
        self.label_folder = label_folder
        self.load_sprite_labels(label_folder)
        if eval_num is None:
            self.eval = False
            self._env = self._create_env(substrate_name)
        else:
            self.eval = True
            self.eval_num = eval_num
            self.substrate_name = substrate_name
            self.scenario_name = substrate_name + f'_{eval_num}'
            self._env = self._create_env(self.scenario_name)
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.sprites)
    
    def get_agent_ids(self) -> List[str]:
        return self._env.get_agent_ids()
    
    def _create_env(self, substrate_name: str):
        if eval:
            # create scenario eval
            env = utils.scenario_creator(substrate_name)
        else:         
            env_config = substrate.get_config(substrate_name)
            self.roles = env_config.default_player_roles
            self.substrate_name = substrate_name
            env = utils.env_creator({
                'substrate': self.substrate_name,
                'roles': self.roles,
            })
        return env

    def load_sprite_labels(self, label_folder: Path):        
        sprites = []
        sprite_labels = []
        for filename in os.listdir(label_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image = np.array(Image.open(os.path.join(label_folder, filename)))
                sprites.append(image.flatten())
                sprite_labels.append(filename.split('.')[0])
        self.sprites = np.array(sprites)
        self.sprite_labels = np.array(sprite_labels)

    def exact_pixel_match(self, patch):
        patch_vector = patch.flatten()
        for ref_img, label in zip(self.sprites, self.sprite_labels):
            if np.array_equal(patch_vector, ref_img):
                return label
        
        return None

    def image_to_state(self, image: np.ndarray):
        """Convert an image to a dictionary of states."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        width, height = image.size
        sprite_size = 8  # sprite size is always 8 for melting pot
        self.width = width
        self.height = height
        self.sprite_size = sprite_size
        
        patch_number = 0
        states = {'global': {}}
        patch_coords = []
        
        timestamp = int(time.time())
        date_time = datetime.fromtimestamp(timestamp)
        timestamp = date_time.strftime("%m%d_%H%M%S")  # Get the current timestamp
        
        for y in range(0, height, sprite_size):
            for x in range(0, width, sprite_size):
                box = (x, y, x + sprite_size, y + sprite_size)
                # crop patches
                patch = np.array(image.crop(box))
                #label_ind = self.knn.kneighbors(patch.flatten().reshape(1, -1), return_distance=False)[0, 0]
                #label = self.sprite_labels[label_ind]
                label = self.exact_pixel_match(patch)
                
                if label is None:
                    # save patch in unlabeled_patches folder
                    patch_img = Image.fromarray(patch)
                    # Get the directory one level up
                    sprite_labels_folder = os.path.dirname(self.label_folder)
                    patch_img.save(f'{sprite_labels_folder}/unlabeled_patches/{timestamp}_{patch_number}.png')
                    image.save(f'{sprite_labels_folder}/unlabeled_patches/unlabeled_frame_{timestamp}_{patch_number}.png')
                    print(f"Patch {patch_number} is not labeled. Please label it and save it in the sprite_labels folder.")

                    label_ind = self.knn.kneighbors(patch.flatten().reshape(1, -1), return_distance=False)[0, 0]
                    label = self.sprite_labels[label_ind]
                    
                    coords = (x // sprite_size, y // sprite_size)
                    patch_coords.append(coords)
                    patch_number += 1
                    
                if label.startswith('wall'):
                    label = 'wall'
                
                coords = (x // sprite_size, y // sprite_size)
                if label in states['global']:
                    states['global'][label].append(coords)
                else:
                    states['global'][label] = [coords]
                
                patch_coords.append(coords)
                patch_number += 1
        
        self.grid = self.build_grid_from_states(states['global'], labels=['wall'])
        return states
    
    def build_grid_from_states(self, states: Dict[str, List[Tuple[int, int]]], labels: List[str], ignore: List[Tuple[int, int]] = None) -> np.ndarray:
        """Build a grid from a dictionary of states. Setting walls and other obstacles you pass in (as str in labels) to 1, 
        and all other things as 0."""
        grid_width = self.width // self.sprite_size
        grid_height = self.height // self.sprite_size
        grid = np.zeros((grid_width, grid_height))
        for label, coords in states.items():
            if label == 'inventory':
                continue
            for x, y in coords:
                if any(label == l or label.startswith(l) for l in labels):
                    grid[x, y] = 1
                else:
                    grid[x, y] = 0

        # Looping through coordinates to ignore (ie. waypoints on plan) and setting the corresponding grid values to 0.0
        if ignore is None:
            ignore = []
            
        for row, col in ignore:
            grid[row, col] = 0.0
        return grid

    def get_ego_state(self, state, player_id): 
        # Extract player's position and orientation
        global_state = state['global']
        for k, v in global_state.items():
            if k.startswith(player_id):
                player_position = v[0]
                player_orientation = k.split('-')[-1]

        # Define the range of the observability window based on orientation
        arena = 'arena' in self.substrate_name
        if arena:
            dims = [11, 11]
        else:
            dims = [5, 5]

        x, y = player_position
        if player_orientation == 'N':
            if arena:
                x_range = range(x - 5, x + 6)
                y_range = range(y - 9, y + 2)
            else:
                x_range = range(x - 2, x + 3)
                y_range = range(y - 3, y + 2)
        elif player_orientation == 'S':
            if arena:
                x_range = range(x - 5, x + 6)
                y_range = range(y - 1, y + 10)
            else:
                x_range = range(x - 2, x + 3)
                y_range = range(y - 1, y + 4)
        elif player_orientation == 'E':
            if arena:
                x_range = range(x - 1, x + 10)
                y_range = range(y - 5, y + 6)
            else:
                x_range = range(x - 1, x + 4)
                y_range = range(y - 2, y + 3)
        elif player_orientation == 'W':
            if arena:
                x_range = range(x - 9, x + 2)
                y_range = range(y - 5, y + 6)
            else:
                x_range = range(x - 3, x + 2)
                y_range = range(y - 2, y + 3)
        else:
            raise ValueError("Invalid player orientation")
        # check that dims are correct
        assert len(x_range) == dims[0] and len(y_range) == dims[1], f"Observability window is not correct. Got {len(x_range)}x{len(y_range)} but expected {dims[0]}x{dims[1]}."

        # Filter the global state to include only entities within the observability window
        egocentric_state = {}
        for entity, positions in global_state.items():
            if entity != player_id:  # Exclude the player themselves from the state
                visible_positions = [pos for pos in positions if pos[0] in x_range and pos[1] in y_range]
                if visible_positions:
                    egocentric_state[entity] = visible_positions

        state[player_id] = egocentric_state

        return state

        
    def reset(self) -> Tuple[Dict[str, List[Tuple[int, int]]], Optional[Dict[str, List[Tuple[int, int]]]]]:
        obs, info = self._env.reset()
        world_rgb = obs['player_0']['WORLD.RGB']
        states = self.image_to_state(world_rgb)
        # add egocentric state and inventory information to states
        for player, player_obs in obs.items():
            states = self.get_ego_state(states, player)
            if 'INVENTORY' in player_obs:
                states[player]['inventory'] = player_obs['INVENTORY']
        return states, obs, info
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, List[Tuple[int, int]]], float, bool, Dict[str, List[Tuple[int, int]]]]:
        obs, rewards, done, truncated, info = self._env.step(actions)
        world_rgb = obs['player_0']['WORLD.RGB']
        states = self.image_to_state(world_rgb)
        # add egocentric state and inventory information to states
        for player, player_obs in obs.items():
            # if no key in states starts with player
            if not any([k.startswith(player) for k in states['global'].keys()]):
                states[player] = {}
            else:
                states = self.get_ego_state(states, player)
            if 'INVENTORY' in player_obs:
                states[player]['inventory'] = player_obs['INVENTORY']
        return states, obs, rewards, done, truncated, info
    

if __name__ == '__main__':
    substrate_name = 'commons_harvest__open'
    sprite_label_folder = '../sprite_labels/commons_harvest__open/'
    env = MeltingPotLLMEnv(substrate_name, sprite_label_folder)
    states, info = env.reset()
    print(states)
    for step in range(1, 15):        
        actions = env._env.action_space_sample()
        states, rewards, _, _, _ = env.step(actions)
        print(f"Step {step}: {states}")
        players = [p for p in states.keys() if p.startswith('player')]
        print(states.keys(), len(players)) 