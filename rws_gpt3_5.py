import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import re
import json
import time
import numpy as np
import pandas as pd
import asyncio
import argparse
import random
import datetime
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
    "FIRE_ZAP": 7,
}

GPT_VERSIONS = {
    '4': 'gpt-4-1106-preview',
    '35': 'gpt-3.5-turbo-1106',
    '125': 'gpt-4-0125-preview'
}


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
    if agent_type == 'v2':
        from llm_plan.agent.tom_modular_agent import DecentralizedAgent
    elif agent_type == 'v3':
        from llm_plan.agent.tom_hypothesis_agent import DecentralizedAgent
        agent_config['self_improve'] = False
    elif agent_type == 'v4':
        from llm_plan.agent.tom_hypothesis_agent import DecentralizedAgent
        agent_config['self_improve'] = True
    agent = DecentralizedAgent(agent_config, controller)    
    return agent

def check_plan_one_step(action, agent, state, env, agent_goals_and_actions):
    # use one step lookahead to see if action takes us to valid/intended location
    next_state_type, new_pos = agent.check_next_state_type(state, action)
    goal_and_plan = agent_goals_and_actions[agent.agent_id]
    subgoal = goal_and_plan['action_plan'][goal_and_plan['subgoal_num']]
    if next_state_type != 'ground' and new_pos != agent.destination and action != 'FIRE_ZAP' and action[:8] != 'INTERACT' and subgoal[:7] != 'fire_at':
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
        # make pathfinding grid include all resources excluding ones on the plan
        # Extracting coordinates from the action plans to exclude from grid
        waypoints = set()
        tuples = re.findall(r'\((\d+,\s*\d+)\)', subgoal)
        for tup in tuples:
            waypoints.add(tuple(map(int, tup.split(','))))
        waypoints = list(waypoints)
        # combine all known states
        agent.combine_all_known_states(state) # update agent.all_known_states
        opponent_key = ['player_1' if agent.agent_id == 'player_0' else 'player_0'][0]
        labels = ['wall', 'yellow_box', 'blue_box', 'purple_box', opponent_key]
        plan_grid = env.build_grid_from_states(agent.all_known_states, labels, waypoints)
        # empty actions queue and get new actions with new pos and new information
        while not agent.all_actions.empty():
            agent.all_actions.get()
        agent.get_actions_from_plan(goal_and_plan, plan_grid, state)
        action = agent.act()

    if action is None:
        print(f"Agent {agent.agent_id} is None, choosing NOOP.")
        action = 'NOOP'

    return action, agent, agent_goals_and_actions

async def main_async(scenario_num: int, agent_type: str, gpt_version: str):
    start_time = time.time()
    manual = False
    """Environment setup""" 
    substrate_name = 'running_with_scissors_in_the_matrix__repeated' 
    sprite_label_path = f'/ccn2/u/locross/llmv_marl/llm_plan/sprite_labels/{substrate_name}'
    env = MeltingPotLLMEnv(substrate_name, sprite_label_path, scenario_num)
    """Agent setup"""
    api_key_path = '/ccn2/u/locross/llmv_marl/llm_plan/lc_api_key.json'
    OPENAI_KEYS = json.load(open(api_key_path, 'r'))
    api_key = OPENAI_KEYS['API_KEY']
    model_settings = {
        "model": GPT_VERSIONS[gpt_version],
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
    frame_folder = f'./frames/{substrate_name}/agent_{agent_type}/gpt_{gpt_version}_scenario_{scenario_num}_{date_time_str}'
    save_obs_frames(obs, step, frame_folder)
    agent.update_state(state)
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
    response_data_list = []
    user_message_data_list = []
    interaction_rewards = {}
    hls_response, subgoal_response, hls_user_msg, subgoal_user_msg = await agent.two_level_plan(state, execution_outcomes, get_action_from_response_errors, reward_tracker, interaction_rewards, step)   
    print(f"Initial response: {hls_response}")
    print('\n')
    print(f"First subgoals: {subgoal_response}")
    with open(output_data_path, 'a') as file:
        file.write(f"ToM Agent, playing scenario {scenario_num}\n\n")
        file.write(f"Initial response: {hls_response}\n\n")
        file.write(f"First subgoals: {subgoal_response}\n\n")
    goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
    valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
    counter = 0
    while not valid_plan and counter < 4:
        print(f"Invalid plan for {agent.agent_id}. Trying again.")
        user_message = agent.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
        plan_response = plan_response + user_message
        responses = await asyncio.gather(
            *[agent.controller.async_batch_prompt(agent.system_message, [plan_response])]
        )
        subgoal_response = responses[0][0]
        goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
        valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
        counter += 1

    print(agent.agent_id, goal_and_plan)
    response_data_list.append(hls_response)
    response_data_list.append(subgoal_response)
    user_message_data_list.append(hls_user_msg)
    user_message_data_list.append(subgoal_user_msg)
    #breakpoint()
    # set which subgoal we are on
    goal_and_plan['subgoal_num'] = 0
    agent_goals_and_actions[agent.agent_id] = goal_and_plan
    # make pathfinding grid include all resources excluding ones on the plan
    # Extracting coordinates from the action plans to exclude from grid
    waypoints = set()
    subgoal = goal_and_plan['action_plan'][goal_and_plan['subgoal_num']]
    tuples = re.findall(r'\((\d+,\s*\d+)\)', subgoal)
    for tup in tuples:
        waypoints.add(tuple(map(int, tup.split(','))))
    waypoints = list(waypoints)
    # combine all known states
    agent.combine_all_known_states(state) # update agent.all_known_states
    opponent_key = 'player_1'
    labels = ['wall', 'yellow_box', 'blue_box', 'purple_box', opponent_key]
    plan_grid = env.build_grid_from_states(agent.all_known_states, labels, waypoints)
    agent.get_actions_from_plan(goal_and_plan, plan_grid, state)

    total_cost = sum(agent.controller.total_inference_cost for agent in agents)
    print(f"Step {step} Total Inference Cost:", total_cost)
    #breakpoint()
    done = False
    during_interaction = False
    during_respawn = False
    after_interaction = False
    #interaction_history = []
    inventory_per_step = []

    while not done:
    #while step < 200:
        step += 1
        """Agent actions"""
        step_actions_dict = {}
        agents_no_actions = []    # keep track agents that are out of actions
        goal_and_plan = agent_goals_and_actions[agent.agent_id]
        action = agent.act()

        # check if subgoal is completed, if so plan actions for the next subgoal
        if not action and len(goal_and_plan['action_plan']) != (goal_and_plan['subgoal_num']+1):
            goal_and_plan['subgoal_num'] += 1
            agent_goals_and_actions[agent.agent_id] = goal_and_plan
            # Extracting coordinates from the action plans to exclude from grid
            waypoints = set()
            subgoal = goal_and_plan['action_plan'][goal_and_plan['subgoal_num']]
            tuples = re.findall(r'\((\d+,\s*\d+)\)', subgoal)
            for tup in tuples:
                waypoints.add(tuple(map(int, tup.split(','))))
            waypoints = list(waypoints)
            # combine all known states
            opponent_key = 'player_1'
            labels = ['wall', 'yellow_box', 'blue_box', 'purple_box', opponent_key]
            plan_grid = env.build_grid_from_states(agent.all_known_states, labels, waypoints)
            agent.get_actions_from_plan(goal_and_plan, plan_grid, state)
            action = agent.act()

        if action and action[:8] == 'INTERACT' or action == 'FIRE_ZAP':
            if action == 'FIRE_ZAP':
                interact_action = action
                next_action = 'INTERACT_'+str(agent.pos)
            else:
                location = action[9:]
                agent.interact(state, location)
                interact_action = agent.act()
                next_action = action
            step_actions_dict[agent.agent_id] = ACTION_IDX[interact_action]
            # count number of steps waiting for interaction, max = 20
            agent.interact_steps += 1
            # if not fire action to start interaction, add action back to buffer
            #if interact_action != 'FIRE_ZAP' and agent.all_actions.qsize() == 0:
            if not during_interaction and not after_interaction and agent.all_actions.qsize() == 0 and agent.interact_steps < 20:
                # put INTERACT action back in to come back here on the next step
                agent.all_actions.put(next_action)
            elif agent.interact_steps >= 20:
                execution_outcomes[agent.agent_id] = 'Been waiting for an interaction at this coordinate for 20 steps. Move to a different location to find opponent.'
            else:
                print(f"HERE IN THE INTERACTION LOOP")
        elif action:
            # use one step lookahead to see if action takes us to valid/intended location
            action, agent, agent_goals_and_actions = check_plan_one_step(action, agent, state, env, agent_goals_and_actions)
            step_actions_dict[agent.agent_id] = ACTION_IDX[action]
        else:
            agents_no_actions.append(agent)

        # API call for agents that are out of actions
        if len(agents_no_actions) > 0 and not during_interaction:
            user_messages = [agent.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)]
            user_message_data_list.append(user_messages)
            # Gathering responses asynchronously
            responses = await asyncio.gather(
                *[agent.controller.async_batch_prompt(agent.system_message, user_messages)]
            )
            subgoal_response = responses[0][0]
            print(f"Response: {subgoal_response}")   
            goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
            # check that this is a valid plan
            valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
            counter = 0
            while not valid_plan and counter < 4:
                print(f"Invalid plan for {agent.agent_id}. Trying again.")
                user_message = agent.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
                plan_response = plan_response + user_message
                responses = await asyncio.gather(
                    *[agent.controller.async_batch_prompt(agent.system_message, [plan_response])]
                )
                subgoal_response = responses[0][0]
                goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
                if len(goal_and_plan) == 0 or len(goal_and_plan['action_plan']) == 0:
                    print(f"Empty goal and plan for {agent.agent_id}")
                    valid_plan = False
                    plan_response = "action_plan in incorrect format. Try again."
                else:
                    valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
                counter += 1

            print(agent.agent_id, goal_and_plan)
            response_data_list.append(subgoal_response)
            user_message_data_list.append(subgoal_user_msg)     
            # set which subgoal we are on
            goal_and_plan['subgoal_num'] = 0              
            agent_goals_and_actions[agent.agent_id] = goal_and_plan                        
            parse_outcome = agent.get_actions_from_plan(goal_and_plan, env.grid, state)
            get_action_from_response_errors[agent.agent_id] = parse_outcome
            action = agent.act()
            if action is None:
                print(f"Agent {agent.agent_id} is None, choosing NOOP.")
                step_actions_dict[agent.agent_id] = 0
            elif action[:8] == 'INTERACT':
                location = action[9:]
                agent.interact(state, location)
                interact_action = agent.act()
                next_action = action
                step_actions_dict[agent.agent_id] = ACTION_IDX[interact_action]
                # count number of steps waiting for interaction, max = 20
                agent.interact_steps += 1
                # put INTERACT action back in to continue interacting when actions are finished
                agent.all_actions.put(next_action)
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
        print(f"Step {step} Total Inference Cost:", total_cost)
        # update memory with currently observed states
        agent.update_memory(state, step)

        """Environment step"""
        if during_interaction:
            # if no player 0, take noop steps until environment is reset
            step_actions_dict = {'player_0': 0}
        state, obs, rewards, done, _, _ = env.step(step_actions_dict)
        current_inventory = state['player_0']['inventory']
        inventory_per_step.append(current_inventory)
        # determine if we are in an interaction
        for key in state['global'].keys():
            if 'inter' in key and not during_interaction:
                during_interaction = True
                current_inventory = state['player_0']['inventory']
                base_inventory = np.array([1., 1., 1.])
                if np.array_equal(base_inventory, current_inventory):
                    # weird bug where inventory is reset to base inventory
                    # Iterate backwards through inventory_per_step to find the most recent different inventory
                    for previous_inventory in reversed(inventory_per_step):
                        if not np.array_equal(base_inventory, previous_inventory):
                            current_inventory = previous_inventory
                            break  # Exit the loop once the most recent different inventory is found
                    print("Weird bug where inventory is reset to base inventory")
                    print(f"Inventory reset to base inventory. Previous inventory: {current_inventory}")
                    #breakpoint()
                interaction_inventory = {
                    'rock/yellow': int(current_inventory[0]), 
                    'paper/purple': int(current_inventory[1]), 
                    'scissors/blue': int(current_inventory[2])
                }
                break
        step_rewards = {agent.agent_id: 0 for agent in agents}
        for agent_id, reward in rewards.items():
            # Round reward to 3 decimal points
            rounded_reward = np.round(reward, 3)
            reward_tracker[agent_id] += rounded_reward
            step_rewards[agent_id] += rounded_reward

        # determine if we are in respawning phase
        reset_env = any(key.startswith('player_0') for key in state['global'])
        if during_interaction and not reset_env and not during_respawn:
            print(f"Interaction: Agent {agent_id}, inventory: {interaction_inventory}, received reward {reward}")
            after_interaction = True 
            interaction_rewards = {agent_id: np.round(reward, 3) for agent_id, reward in rewards.items()}
            during_respawn = True

        done = done['__all__']
        save_obs_frames(obs, step, frame_folder)
        # print(f"Step {step} rewards: {step_rewards}, Player 0 inventory: {state['player_0']['inventory']}")
        print(f"Step {step} rewards: {reward_tracker}, Player 0 inventory: {state['player_0']['inventory']}")
        if after_interaction and reset_env:
            during_interaction = False
            during_respawn = False
            agent.interaction_num += 1
            interaction_dict = {
                "Interaction": agent.interaction_num,
                "your_inventory": interaction_inventory,
                "rewards": interaction_rewards[agent.agent_id].item()
            }
            agent.interaction_history.append(interaction_dict)
            # give feedback and reset all plans
            hls_response, subgoal_response, hls_user_msg, subgoal_user_msg = await agent.two_level_plan(state, execution_outcomes, get_action_from_response_errors, reward_tracker, interaction_rewards, step, after_interaction=True)
            print(f"Interaction {agent.interaction_num}")
            print(f"Step: {step}, Total reward: {reward_tracker}")
            print(f"Interaction {agent.interaction_num}: inventory={interaction_inventory}, rewards={interaction_rewards[agent.agent_id].item()}")
            print('\n')
            print(f"HLS: {hls_response}")
            print('\n')
            print(f"Next subgoals: {subgoal_response}")
            with open(output_data_path, 'a') as file:
                file.write(f"Interaction {agent.interaction_num}, inventory={interaction_inventory}, rewards={interaction_rewards[agent.agent_id].item()}\n")
                file.write(f"{agent.interaction_history}\n")
                file.write(f"Step: {step}, Total reward: {reward_tracker}\n\n")
                file.write(f"HLS: {hls_response}\n\n")
                file.write(f"Next subgoals: {subgoal_response}\n\n")
            #breakpoint()
            response_data_list.append(hls_response)
            response_data_list.append(subgoal_response)
            # Empty the action queue
            while not agent.all_actions.empty():
                agent.all_actions.get()
            goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
            # check that this is a valid plan
            valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
            counter = 0
            while not valid_plan and counter < 4:
                print(f"Invalid plan for {agent.agent_id}. Trying again.")
                user_message = agent.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
                plan_response = plan_response + user_message
                responses = await asyncio.gather(
                    *[agent.controller.async_batch_prompt(agent.system_message, [plan_response])]
                )
                response = responses[0][0]  
                goal_and_plan = agent.extract_goals_and_actions(response)
                if len(goal_and_plan) == 0 or len(goal_and_plan['action_plan']) == 0:
                    print(f"Empty goal and plan for {agent.agent_id}")
                    valid_plan = False
                    plan_response = "action_plan in incorrect format. Try again."
                else:
                    valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
                counter += 1
            print(agent.agent_id, goal_and_plan)
            # set which subgoal we are on
            goal_and_plan['subgoal_num'] = 0               
            agent_goals_and_actions[agent.agent_id] = goal_and_plan                        
            parse_outcome = agent.get_actions_from_plan(goal_and_plan, env.grid, state)
            get_action_from_response_errors[agent.agent_id] = parse_outcome
            after_interaction = False
        # keep track of the execution outcomes
        if not during_interaction:
            execution_outcomes = {}
            execution_outcomes[agent.agent_id] =  agent.update_state(state)
    print(f"Episode finished at step {step} with rewards {reward_tracker}")

    # save results in minutes
    total_duration = time.time() - start_time
    total_duration = total_duration / 60
    # make dataframe - columns for agent_type, scenario, reward, datetime
    df_results = pd.DataFrame({'agent_type': [agent_type], 'scenario': [scenario_num], 
                               'reward': [reward_tracker['player_0']], 'steps': [step], 
                               'duration': [total_duration], 'cost': [total_cost], 
                               'datetime': [date_time_str], 'gpt_version': [GPT_VERSIONS[gpt_version]]})
    all_results_file = './results/rws_scores.csv'
    if os.path.exists(all_results_file):
        df_all_results = pd.read_csv(all_results_file, index_col=0)
        df_all_results = pd.concat([df_all_results, df_results], ignore_index=True)
    else:
        df_all_results = df_results
    df_all_results.to_csv(all_results_file)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run main_async from specified file with scenario number.')

    # Add the arguments
    parser.add_argument('--agent', 
                        type=str, 
                        choices=['v1', 'v2', 'v3', 'v4'],
                        help='The agent to run: "tom" for rws_tom_agent or "hierarchical" for rws_eval')

    parser.add_argument('--scenario_num', 
                        type=int, 
                        help='The scenario number to pass into main_async')

    parser.add_argument('--gpt_version', 
                        type=str, 
                        default='35',
                        choices=['4', '35', '125'],
                        help='The GPT version to use')

    parser.add_argument('--num_seeds', 
                        type=int, 
                        default=1,
                        help='The number of seeds to run for each scenario')

    # Execute the parse_args() method
    args = parser.parse_args()

    for i in range(args.num_seeds):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_async(args.scenario_num, args.agent, args.gpt_version))


if __name__ == "__main__":
    main()