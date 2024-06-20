import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import json
import random
import datetime
from PIL import Image
from meltingpot import substrate
from typing import Dict, List, Tuple, Optional, Any

from llm_plan.agent.agent import Agent
from llm_plan.env.mp_llm_env import MeltingPotLLMEnv
from llm_plan.controller.gpt_controller_rws import GPTController
import llm_plan.env.utils as utils


ACTION_IDX = {
    "NOOP": 0,
    "FORWARD": 1,
    "BACKWARD": 2,
    "STEP_LEFT": 3,
    "STEP_RIGHT": 4,    
    "TURN_LEFT": 5,
    "TURN_RIGHT": 6,
    "FIRE_ZAP": 7,
}

def equalize_action_lengths(action_dict):
    """Pad shorter action lists with NOOPs."""
    max_length = max(len(actions) for actions in action_dict.values())    
    for player, actions in action_dict.items():
        while len(actions) < max_length:
            actions.append('NOOP')
    return max_length


def save_obs_frames(obs: Dict[str, Any], step: int, frame_folder: str):        
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


def compute_api_cost(usage_data, completion_cost=0.03, prompt_cost=0.001):
    return (usage_data['completion_tokens'] * completion_cost \
        + usage_data['prompt_tokens'] * prompt_cost)


def main():
    manual = False
    # substrate and image2states converter
    #substrate_name = 'commons_harvest__open'   
    substrate_name = 'running_with_scissors_in_the_matrix__repeated' 
    sprite_label_path = f'/ccn2/u/locross/llmv_marl/llm_plan/sprite_labels/{substrate_name}'
    env = MeltingPotLLMEnv(substrate_name, sprite_label_path)
    # env_config = substrate.get_config(substrate_name)
    # roles = env_config.default_player_roles
    # substrate_name = substrate_name
    # env = utils.env_creator({
    #     'substrate': substrate_name,
    #     'roles': roles,
    # })
    # initialize all agents
    agents = []
    agent_ids = env.get_agent_ids()
    # sort
    agent_ids = sorted(list(agent_ids))
    agent_configs = [{'agent_id':  agent_id} for agent_id in agent_ids]
    for agent_config in agent_configs:
        agents.append(Agent(agent_config))
    # gpt controller
    api_key_path = '/ccn2/u/locross/llmv_marl/llm_plan/lc_api_key.json'
    OPENAI_KEYS = json.load(open(api_key_path, 'r'))
    controller = GPTController(OPENAI_KEYS['API_KEY'])
    # main loop (reset is step 0)    
    step = 0
    state, obs, _ = env.reset()
    # @TODO: this should be part of the config
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Create the subfolder path
    frame_folder = f'./frames/{substrate_name}/{date_time_str}'
    save_obs_frames(obs, step, frame_folder)
    for agent in agents:
        agent.update_state(state)
    user_prompt = controller.generate_initial_user_message(state)
    if manual:
        goals_and_actions, response_data = controller.get_actions_manually(state, user_prompt)  
    else: 
        goals_and_actions, response_data = controller.get_actions_from_gpt4(user_prompt)
        total_cost = compute_api_cost(response_data['usage'])
        print(f"Current api cost: {total_cost}")
    breakpoint()
    done = False
    # keep track of response from APT calls and save it 
    response_data_list = [response_data]  # Initialize list to store response data
    response_data_path = os.path.join(frame_folder, 'response_data.json')
    reward_tracker = {agent.agent_id: 0 for agent in agents}
    reward_tracker['collective_reward'] = 0
    #while not done:
    while step < 100:
        step_actions = {}
        for agent in agents:
            agent_goal_and_plan = goals_and_actions[agent.agent_id]    
            print(agent.agent_id, agent_goal_and_plan)
            action = agent.get_actions_from_plan(agent_goal_and_plan, env.grid)
            step_actions[agent.agent_id] = action   
        num_steps = equalize_action_lengths(step_actions) 
        step_rewards = {agent.agent_id: 0 for agent in agents}
        for action_step in range(0, num_steps):
            step += 1
            actions = {
                agent.agent_id: ACTION_IDX[step_actions[agent.agent_id][action_step]] 
                for agent in agents if agent.agent_id in step_actions
                }
            state, obs, rewards, done, truncated, info = env.step(actions)
            done = done['__all__']
            save_obs_frames(obs, step, frame_folder)
            for agent in agents:
                step_rewards[agent.agent_id] += rewards[agent.agent_id]
        # actions = {
        #         agent.agent_id: ACTION_IDX[random.choice(list(ACTION_IDX.keys()))] 
        #         for agent in agents
        #         }
        #state, obs, rewards, done, truncated, info = env.step(actions)
        #obs, rewards, done, truncated, info = env.step(actions)
        #done = done['__all__']
        #save_obs_frames(obs, step, frame_folder)
        # for agent in agents:
        #     step_rewards[agent.agent_id] += rewards[agent.agent_id]
        # step_rewards['collective_reward'] += obs['player_0']['COLLECTIVE_REWARD']
        # update reward tracker
        # keep track of the outcome of action plan execution
        execution_outcomes = {}
        for agent in agents:
            execution_outcomes[agent.agent_id] = agent.update_state(state)                    
        user_prompt = controller.generate_feedback_user_message(
            state, execution_outcomes, rewards)
        if manual:
            goals_and_actions, response_data = controller.get_actions_manually(state, user_prompt)
        else:
            goals_and_actions, response_data = controller.get_actions_from_gpt4(user_prompt)
            response_data_list.append(response_data)
            api_cost = compute_api_cost(response_data['usage'])
            print(f"The api call cost {api_cost}")
            total_cost += api_cost
            print(f"Current total api cost: {total_cost}")
        for agent in agents:
            reward_tracker[agent.agent_id] += step_rewards[agent.agent_id]
        print(f"Step {step} rewards: {step_rewards}")
        with open(response_data_path, 'w') as file:
            data = {
                'response_data': response_data_list, 
                'reward_tracker': reward_tracker
                }
            json.dump(data, file, indent=4)  



if __name__ == "__main__":
    main()