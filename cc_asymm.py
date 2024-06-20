import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import re
import json
import time
import numpy as np
import pandas as pd
import argparse
import asyncio
import random
import datetime
import subprocess
from PIL import Image
from meltingpot import substrate
from typing import Dict, List, Tuple, Optional, Any

from llm_plan.env.mp_llm_env import MeltingPotLLMEnv
from llm_plan.controller.async_llm import AsyncChatLLM
from llm_plan.controller.async_gpt_controller import AsyncGPTController


ACTION_IDX = {
    "NOOP": 0,
    "FORWARD": 1,
    "BACKWARD": 2,
    "STEP_LEFT": 3,
    "STEP_RIGHT": 4,    
    "TURN_LEFT": 5,
    "TURN_RIGHT": 6,
    "INTERACT": 7,
}


def print_and_save(*args, new_line=True, **kwargs):
    global all_output_file
    print(*args, **kwargs)
    with open(all_output_file, 'a') as file:
        print(*args, **kwargs, file=file)
        if new_line:
            print('\n', file=file)


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


def setup_agent(api_key, model_id, model_settings, agent_type):
    llm = AsyncChatLLM(api_key=api_key)
    controller = AsyncGPTController(
        llm=llm,
        model_id=model_id,
        **model_settings
    )
    agent_config = {'agent_id': model_id}
    if agent_type == 'reflexion':
        from llm_plan.agent.cc_reflexion import DecentralizedAgent
    elif agent_type == 'react':
        from llm_plan.agent.cc_react import DecentralizedAgent
    elif agent_type == 'v2':
        from llm_plan.agent.cc_modular_agent import DecentralizedAgent
    elif agent_type == 'v3':
        from llm_plan.agent.cc_hypothetical_minds import DecentralizedAgent
        agent_config['self_improve'] = False
    elif agent_type == 'hm':
        from llm_plan.agent.cc_hypothetical_minds import DecentralizedAgent
        agent_config['self_improve'] = True
    elif agent_type == 'hypothesis':
        from llm_plan.agent.hypothesis_only_agent import DecentralizedAgent
        agent_config['self_improve'] = True
    agent = DecentralizedAgent(agent_config, controller)    
    return agent

def check_plan_one_step(action, agent, state, env, agent_goals_and_actions):
    # use one step lookahead to see if action takes us to valid/intended location
    next_state_type, new_pos = agent.check_next_state_type(state, action)
    goal_and_plan = agent_goals_and_actions[agent.agent_id]
    subgoal = goal_and_plan['action_plan'][goal_and_plan['subgoal_num']]
    if next_state_type != 'ground' and new_pos != agent.destination and subgoal[:8] != 'interact':
        # if next state is not ground, ie. collects unintended resource, replan with newly observed state information
        # update current subgoal with current position
        subgoal = goal_and_plan['action_plan'][goal_and_plan['subgoal_num']]
        # Splitting the subgoal into two parts at the first closing parenthesis
        part1, part2 = subgoal.split('),', 1)
        # Updating the first part with the agent's current position
        updated_part1 = part1[:part1.find('(') + 1] + str(agent.current_pos)
        # Reassembling the updated subgoal
        subgoal = updated_part1 + ',' + part2
        goal_and_plan['action_plan'][goal_and_plan['subgoal_num']] = subgoal
        agent_goals_and_actions[agent.agent_id] = goal_and_plan
        plan_grid = build_grid_from_states(state)
        # empty actions queue and get new actions with new pos and new information
        while not agent.all_actions.empty():
            agent.all_actions.get()
        agent.get_actions_from_plan(goal_and_plan, plan_grid, state)
        action = agent.act()

    if action is None:
        print(f"Agent {agent.agent_id} is None, choosing NOOP.")
        action = 'NOOP'

    return action, agent, agent_goals_and_actions


def build_grid_from_states(state):
    """Build a grid from a dictionary of state. Setting walls and other obstacles you pass in (as str in labels) to 1, 
    and all other things as 0."""
    grid_width = 9
    grid_height = 5
    grid = np.ones((grid_width, grid_height))
    player_x = [v for k, v in state['global'].items() if k.startswith('player_0')][0][0][0]
    # player is on left of kitchen if x is < 4 and on right if x is > 4
    player_spot = 'left' if player_x < 4 else 'right'

    for label, coords in state['global'].items():
        if label == 'ground':
            for x, y in coords:
                if player_spot == 'left' and x > 4:
                    continue
                elif player_spot == 'right' and x < 4:
                    continue
                else:
                    grid[x, y] = 0

        if label.startswith('player_0'):
            for x, y in coords:
                grid[x, y] = 0

    return grid


def wait_for_pot(agent, action, steps_waiting, wait_thr=25):
    # take noops until pot is done
    coord_str = action.split('((', 1)[1].split(')')[0]
    pot_loc = tuple(map(int, coord_str.split(', ')))
    if not agent.pot_state[pot_loc]["done"] and steps_waiting < wait_thr:
        # put action back in to come back here on the next step
        agent.all_actions.put(action)
        action = 'NOOP'
        steps_waiting += 1
    else:
        # pot is done, go to next subgoal
        action = None
        steps_waiting = 0

    return action, steps_waiting


async def main_async(scenario_num: int, agent_type: str):
    start_time = time.time()
    manual = False
    """Environment setup""" 
    substrate_name = 'collaborative_cooking__asymmetric' 
    sprite_label_path = f'/ccn2/u/locross/llmv_marl/llm_plan/sprite_labels/{substrate_name}'
    env = MeltingPotLLMEnv(substrate_name, sprite_label_path, scenario_num)
    """Agent setup"""
    api_key_path = '/ccn2/u/locross/llmv_marl/llm_plan/lc_api_key.json'
    OPENAI_KEYS = json.load(open(api_key_path, 'r'))
    api_key = OPENAI_KEYS['API_KEY']
    model_settings = {
        "model": "gpt-4-1106-preview",
        "max_tokens": 4000,
        "temperature": 0.1,
        "top_p": 1.0,
        "n": 1,
    }
    agent = setup_agent(api_key, model_id=f"player_0", model_settings=model_settings, agent_type=agent_type)
    """Interaction with the environment"""
    # initial step    
    step = 0
    state, obs, _ = env.reset()
    # @TODO: this should be part of the config
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Create the subfolder path
    frame_folder = f'./frames/{substrate_name}/agent_{agent_type}/scenario_{scenario_num}_{date_time_str}'
    save_obs_frames(obs, step, frame_folder)
    agent.update_state(state, step)
    agent_goals_and_actions = {}    
    # Generate subgoals to execute given high-level strategy
    # keep track of the execution outcomes
    execution_outcomes = {}
    # keep track of agents that have errors while parsing response to get actions
    get_action_from_response_errors = {}
    # track rewards and responses
    agents = [agent]
    reward_tracker = {agent.agent_id: 0} 
    output_data_path = os.path.join(frame_folder, 'output_data.txt')
    response_data_path = os.path.join(frame_folder, 'response_data.json')
    global all_output_file
    all_output_file = os.path.join(frame_folder, 'all_output_data.txt')
    with open(all_output_file, 'w') as file:
        pass
    response_data_list = []
    user_message_data_list = []
    state_info_dict_list = []
    state_info_dict = agent.get_state_info(state, step)
    state_info_dict_list.append(state_info_dict)
    hls_response, subgoal_response, hls_user_msg, subgoal_user_msg = await agent.two_level_plan(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)   
    print_and_save(f"Initial response: {hls_response}")
    print_and_save('\n')
    print_and_save(f"First subgoals: {subgoal_response}")
    with open(output_data_path, 'a') as file:
        file.write(f"ToM Agent, playing scenario {scenario_num}\n\n")
        file.write(f"Initial response: {hls_response}\n\n")
        file.write(f"First subgoals: {subgoal_response}\n\n")
    goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
    valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
    counter = 0
    while not valid_plan and counter < 10:
        #breakpoint()
        print_and_save(f"Invalid plan for {agent.agent_id}, {plan_response}. Trying again.")
        evaluator_feedback = ''
        user_message = agent.generate_feedback_user_message(state, plan_response, get_action_from_response_errors, evaluator_feedback, reward_tracker, step)
        plan_response = plan_response + user_message
        responses = await asyncio.gather(
            *[agent.controller.async_batch_prompt(agent.system_message, [plan_response])]
        )
        subgoal_response = responses[0][0]
        goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
        valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
        counter += 1

    print_and_save(agent.agent_id, goal_and_plan)
    response_data_list.append(hls_response)
    response_data_list.append(subgoal_response)
    user_message_data_list.append(hls_user_msg)
    user_message_data_list.append(subgoal_user_msg)
    # set which subgoal we are on
    goal_and_plan['subgoal_num'] = 0
    agent_goals_and_actions[agent.agent_id] = goal_and_plan
    plan_grid = build_grid_from_states(state)
    #breakpoint()
    agent.get_actions_from_plan(goal_and_plan, plan_grid, state)

    total_cost = sum(agent.controller.total_inference_cost for agent in agents)
    print_and_save(f"Step {step} Total Inference Cost:", total_cost, new_line=False)
    done = False
    steps_waiting = 0
    reward_during_plan = 0 
    dish_delivered = False

    while not done:
    #while step < 200:
        step += 1
        """Agent actions"""
        step_actions_dict = {}
        agents_no_actions = []    # keep track agents that are out of actions
        goal_and_plan = agent_goals_and_actions[agent.agent_id]
        action = agent.act()

        if action and action[:4] == 'wait':
            print('Steps waiting', steps_waiting)
            #breakpoint()
            action, steps_waiting = wait_for_pot(agent, action, steps_waiting)

        # check if subgoal is completed, if so plan actions for the next subgoal
        if not action and len(goal_and_plan['action_plan']) != (goal_and_plan['subgoal_num']+1):
            goal_and_plan['subgoal_num'] += 1
            agent_goals_and_actions[agent.agent_id] = goal_and_plan
            agent.get_actions_from_plan(goal_and_plan, plan_grid, state)
            action = agent.act()

        if action:
            # use one step lookahead to see if action takes us to valid/intended location
            if action[:4] != 'wait' and action != 'NOOP': 
                action, agent, agent_goals_and_actions = check_plan_one_step(action, agent, state, env, agent_goals_and_actions)
            elif action[:4] == 'wait':
                action, steps_waiting = wait_for_pot(agent, action, steps_waiting)
            if action is None:
                print(f"Agent {agent.agent_id} is None, choosing NOOP.")
                action = 'NOOP'
            step_actions_dict[agent.agent_id] = ACTION_IDX[action]
        else:
            agents_no_actions.append(agent)

        if dish_delivered:
            agent.delivery_num += 1
            hls_response, hls_user_msg, teammate_hypothesis = await agent.tom_module(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step) 
            print_and_save(f"ToM module user message: {hls_user_msg}")
            print_and_save(f"Teammate hypothesis: {teammate_hypothesis}")
            print_and_save(f"Response: {hls_response}")
            user_message_data_list.append(hls_user_msg)
            #breakpoint()

        # API call for agents that are out of actions
        if len(agents_no_actions) > 0:
            state_info_dict = agent.get_state_info(state, step)
            state_info_dict_list.append(state_info_dict) 
            user_message_outcomes, evaluator_response = await agent.evaluate_action_outcomes(goal_and_plan, state_info_dict_list, reward_during_plan)
            evaluator_feedback = user_message_outcomes + evaluator_response
            user_message_data_list.append(user_message_outcomes)
            user_message = agent.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, evaluator_feedback, reward_tracker, step)
            user_message_data_list.append(user_message)
            print_and_save(f"User message: {user_message}")
            # Gathering responses asynchronously
            responses = await asyncio.gather(
                *[agent.controller.async_batch_prompt(agent.system_message, [user_message])]
            )
            subgoal_response = responses[0][0]
            print_and_save(f"Response: {subgoal_response}")   
            goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
            # check that this is a valid plan
            valid_plan, feedback_message = agent.is_valid_plan(state, goal_and_plan)
            counter = 0
            while not valid_plan and counter < 10:
                #breakpoint()
                print_and_save(f"Invalid plan for {agent.agent_id}. Trying again.")
                user_message = agent.generate_feedback_user_message(state, feedback_message, get_action_from_response_errors, evaluator_feedback, reward_tracker, step)
                plan_response = plan_response + user_message
                responses = await asyncio.gather(
                    *[agent.controller.async_batch_prompt(agent.system_message, [plan_response])]
                )
                subgoal_response = responses[0][0]
                goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
                if len(goal_and_plan) == 0 or len(goal_and_plan['action_plan']) == 0:
                    print_and_save(f"Empty goal and plan for {agent.agent_id}")
                    valid_plan = False
                    plan_response = "action_plan in incorrect format. Try again."
                else:
                    valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
                #breakpoint()
                counter += 1

            print_and_save(agent.agent_id, goal_and_plan)
            response_data_list.append(subgoal_response)
            user_message_data_list.append(subgoal_user_msg) 
            reward_during_plan = 0   
            #breakpoint()
            # set which subgoal we are on
            goal_and_plan['subgoal_num'] = 0              
            agent_goals_and_actions[agent.agent_id] = goal_and_plan                        
            parse_outcome = agent.get_actions_from_plan(goal_and_plan, plan_grid, state)
            get_action_from_response_errors[agent.agent_id] = parse_outcome
            action = agent.act()
            if action and action[:4] == 'wait':
                action, steps_waiting = wait_for_pot(agent, action, steps_waiting)
                if action is None:
                    print(f"Agent {agent.agent_id} is None, choosing NOOP.")
                    action = 'NOOP'
                step_actions_dict[agent.agent_id] = ACTION_IDX[action]
            elif action is None:
                print(f"Agent {agent.agent_id} is None, choosing NOOP.")
                action = 'NOOP'
                step_actions_dict[agent.agent_id] = ACTION_IDX[action]
            else:
                # use one step lookahead to see if action takes us to valid/intended location
                action, agent, agent_goals_and_actions = check_plan_one_step(action, agent, state, env, agent_goals_and_actions)
                step_actions_dict[agent.agent_id] = ACTION_IDX[action]
        # reset agents_no_actions
        agents_no_actions = []
        # save prompt and responses
        with open(response_data_path, 'w') as file:
            data = {
                'user_messages': user_message_data_list,
                'response_data': response_data_list, 
                'reward_tracker': reward_tracker
                }
            json.dump(data, file, indent=4)
        total_cost = sum(agent.controller.total_inference_cost for agent in agents)
        print_and_save(f"Step {step} Total Inference Cost:", total_cost, new_line=False)
        # update memory with currently observed states
        print(state['player_0'])
        print(action)
        print('\n')
        agent.update_memory(state, step, action)

        """Environment step"""
        if step > 25:
            print('action', action, step_actions_dict)
            #breakpoint()
        state, obs, rewards, done, _, _ = env.step(step_actions_dict)
        step_rewards = {agent.agent_id: 0 for agent in agents}
        for agent_id, reward in rewards.items():
            # Round reward to 3 decimal points
            rounded_reward = np.round(reward, 3)
            reward_tracker[agent_id] += rounded_reward
            step_rewards[agent_id] += rounded_reward
            reward_during_plan += rounded_reward
        # dish delivered if reward is nonzero
        if step_rewards['player_0'] > 0:
            dish_delivered = True
        else:
            dish_delivered = False
        done = done['__all__']
        save_obs_frames(obs, step, frame_folder)
        print_and_save(f"Step {step} rewards: {reward_tracker}", new_line=False)
        # keep track of the execution outcomes
        execution_outcomes = {}
        execution_outcomes[agent.agent_id] =  agent.update_state(state, step)

    print_and_save(f"Episode finished at step {step} with rewards {reward_tracker}")

    # save results in minutes
    total_duration = time.time() - start_time
    total_duration = total_duration / 60
    # make dataframe - columns for agent_type, scenario, reward, datetime
    df_results = pd.DataFrame({'agent_type': [agent_type], 'scenario': [scenario_num], 
                               'reward': [reward_tracker['player_0']], 'steps': [step], 
                               'duration': [total_duration], 'cost': [total_cost], 'datetime': [date_time_str]})
    all_results_file = './results/cc_scores.csv'
    if os.path.exists(all_results_file):
        df_all_results = pd.read_csv(all_results_file, index_col=0)
        df_all_results = pd.concat([df_all_results, df_results], ignore_index=True)
    else:
        df_all_results = df_results
    df_all_results.to_csv(all_results_file)

     # Save video of the frames
    create_video_script = './create_videos.sh'
    subprocess.call([create_video_script, frame_folder])


def main():
    parser = argparse.ArgumentParser(description='Run the Collaborative Cooking agent.')
    parser.add_argument('--scenario_num', type=int, default=0, help='Scenario number')
    parser.add_argument('--agent_type', type=str, default='v4', help='Agent type')
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_async(args.scenario_num, args.agent_type))


if __name__ == "__main__":
    main()