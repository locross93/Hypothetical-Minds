import os
from PIL import Image


def save_obs_frames(obs, step, frame_folder):        
    # save world rgb
    global_frame_folder = os.path.join(frame_folder, 'global')
    if not os.path.exists(global_frame_folder):
        os.makedirs(global_frame_folder)
    world_rgb = obs['player_0']['WORLD.RGB']
    world_rgb = Image.fromarray(world_rgb)
    world_rgb.save(os.path.join(global_frame_folder, f'world_{step}.png'))
    for agent_id, agent_obs in obs.items():
        agent_frame_folder = os.path.join(frame_folder, agent_id)
        if not os.path.exists(agent_frame_folder):
            os.makedirs(agent_frame_folder)
        agent_rgb = agent_obs['RGB']
        agent_rgb = Image.fromarray(agent_rgb)
        agent_rgb.save(os.path.join(agent_frame_folder, f'{agent_id}_{step}.png'))