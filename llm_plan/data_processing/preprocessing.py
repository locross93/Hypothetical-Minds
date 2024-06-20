"""Preprocesses the substrate observations into patches for manual annotation.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from meltingpot import substrate
from meltingpot.utils.substrates import colors
from PIL import Image, ImageDraw
#import env.utils as utils
import llm_plan.env.utils as utils

class MPSubstratesPreprocessor():
  def __init__(self, substrate_name, output_folder, scale_factor=5):
      self.substrate_name = substrate_name
      env_config = substrate.get_config(substrate_name)
      roles = env_config.default_player_roles
      self._num_players = len(roles)
      self._env = utils.env_creator({
          'substrate': substrate_name,
          'roles': roles,
      })      
      self.output_folder = output_folder
      self.scale_factor = scale_factor
      self.get_agent_colors()
      self.video_frames = []

  def get_agent_colors(self):
    """Get colors for each agent."""
    self.agent_colors = {}
    for agent_id in self._env.get_agent_ids():
      agent_idx = int(agent_id.split('_')[-1])
      #self.agent_colors[agent_id] = colors.human_readable[agent_idx]
      self.agent_colors[agent_id] = colors.human_readable[agent_idx+1]
    output_folder = os.path.join(self.output_folder, self.substrate_name)
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
  
    # create a figure with agents and their colors
    fig, axes = plt.subplots(
       1, len(self.agent_colors), figsize=(len(self.agent_colors) * 2, 2))
    # Plot each color square and label them
    for ax, (agent_id, color) in zip(axes, self.agent_colors.items()):
        ax.imshow([[color]], aspect='auto')
        ax.set_title(agent_id)
        ax.axis('off')
    # Adjust layout and save the figure
    plt.tight_layout()
    #plt.savefig(os.path.join(output_folder, 'agent_color.jpg'))    
    plt.savefig(os.path.join(output_folder, 'agent_color.png'))  

  def run_step(self):
    """Test step() returns rewards for all agents."""
    obs, _ = self._env.reset()    
    actions = {}
    # change the range depending on environment
    for step in range(1, 100):
      if step < 5:
        # turn left 4 times to get all orientations for the frist 4 frames 
        for player_idx in range(0, self._num_players):
          actions['player_' + str(player_idx)] = 5
      else:
        # take random action so I can see all possible situations
         for player_idx in range(0, self._num_players):
          #actions['player_' + str(player_idx)] = np.random.randint(1, 8)
          actions = self._env.action_space.sample()

      base_output_folder = os.path.join(
          self.output_folder, self.substrate_name, f'step_{step}')
      if not os.path.exists(base_output_folder):
          os.makedirs(base_output_folder)
      obs, rewards, _, _, _ = self._env.step(actions)
      world_rgb = obs['player_0']['WORLD.RGB']
      patch_coords = self.split_obs_into_patches(world_rgb, base_output_folder)

      # Append the current frame to the video_frames list
      self.video_frames.append(world_rgb)

    # Generate the video from the collected frames
    self.generate_video()

  def split_obs_into_patches(self, image, base_output_folder, sprite_size=8):
      """
      Split an image into patches of individual sprites. Scaling is used to
      make manual annotation easier. But for patch comparison during text generation
      the original patches are used.
      """
      def split_image_into_patches(image, patch_size, output_folder):
        width, height = image.size
        patch_coords = []
        patch_number = 0
        for y in range(0, height, patch_size):
          for x in range(0, width, patch_size):
              box = (x, y, x + patch_size, y + patch_size)
              # crop patches
              patch = image.crop(box)
              #patch_save_path = f"{output_folder}/patch_{patch_number}.jpg"
              patch_save_path = f"{output_folder}/patch_{patch_number}.png"
              patch.save(patch_save_path)
              patch_coords.append((x, y))
              patch_number += 1
        return patch_coords

      # output original and scaled patches into separate folders
      output_folder = os.path.join(
        base_output_folder, 'patches')
      scaled_folder = os.path.join(
        base_output_folder, 'scaled_patches')
      if not os.path.exists(output_folder):
          os.makedirs(output_folder)
          os.makedirs(scaled_folder)
      # split the image into patches
      image = Image.fromarray(image)
      patch_coords = split_image_into_patches(image, sprite_size, output_folder)
      # split resized image into patches
      patch_size = sprite_size * self.scale_factor
      scaled_image = image.resize(
         (image.size[0] * self.scale_factor, image.size[1] * self.scale_factor))
      scaled_patch_coords = split_image_into_patches(
         scaled_image, patch_size, scaled_folder)
      # draw patch numbers on scaled image
      draw = ImageDraw.Draw(scaled_image)
      font_size = 10  # Adjust as needed
      for num, (x, y) in enumerate(patch_coords):
          draw.text(
             (x * self.scale_factor, y * self.scale_factor),
             str(num), fill="white", font_size=font_size)
      step = base_output_folder.split('/')[-1]
      #scaled_image.save(f"{base_output_folder}/{step}.jpg")
      scaled_image.save(f"{base_output_folder}/{step}.png")
      return patch_coords

  def generate_video(self):
        video_output_folder = os.path.join(self.output_folder, self.substrate_name)
        video_output_path = os.path.join(video_output_folder, 'gameplay_video.mp4')

        # Get the dimensions of the video frames
        height, width, _ = self.video_frames[0].shape

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10  # Adjust the fps as needed
        video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

        # Write the frames to the video
        for frame in self.video_frames:
            # Convert the frame from RGB to BGR (OpenCV uses BGR format)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        # Release the VideoWriter object
        video_writer.release()

        print(f"Generated gameplay video: {video_output_path}")


if __name__ == '__main__':
    #substrate_name = 'running_with_scissors_in_the_matrix__arena'
    substrate_name = 'prisoners_dilemma_in_the_matrix__arena'
    #substrate_name = 'prisoners_dilemma_in_the_matrix__repeated'
    #substrate_name = 'collaborative_cooking__asymmetric'
    output_folder = '/ccn2/u/locross/llmv_marl/llm_plan/sprite_labels/unlabeled_patches'
    scale_factor = 5
    preprocessor = MPSubstratesPreprocessor(substrate_name, output_folder, scale_factor)
    preprocessor.run_step()
