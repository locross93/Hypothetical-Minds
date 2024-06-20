import os
import re
import json
import time
import asyncio
import datetime
import numpy as np
import pandas as pd

from utils import save_obs_frames

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

def print_and_save(*args, new_line=True, **kwargs):
    global all_output_file
    print(*args, **kwargs)
    with open(all_output_file, 'a') as file:
        print(*args, **kwargs, file=file)
        if new_line:
            print('\n', file=file)

def make_plan_grid(goal_and_plan, env, agent):
    # Extracting coordinates from the action plans to exclude from grid
    waypoints = set()
    subgoal = goal_and_plan['action_plan'][goal_and_plan['subgoal_num']]
    tuples = re.findall(r'\((\d+,\s*\d+)\)', subgoal)
    for tup in tuples:
        waypoints.add(tuple(map(int, tup.split(','))))
    waypoints = list(waypoints)
    labels = ['wall', 'yellow_box', 'blue_box', 'purple_box']
    for opponent_num in range(1, 8):
        opponent_key = f'player_{opponent_num}'
        labels.append(opponent_key)
    plan_grid = env.build_grid_from_states(agent.all_known_states, labels, waypoints)

    return plan_grid

async def run_episode(env, agent):
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    if agent.llm_type == 'gpt4':
        agent_label = agent.agent_type
    else:
        agent_label = agent.agent_type + '_'+ agent.llm_type
    frame_folder = f'./frames/{env.substrate_name}/agent_{agent_label}/scenario_{env.eval_num}_{date_time_str}'
    # initial step  
    start_time = time.time()  
    step = 0
    state, obs, _ = env.reset()
    save_obs_frames(obs, step, frame_folder)
    agent.update_state(state)
    agent_goals_and_actions = {}    
    # Generate subgoals to execute given high-level strategy
    # keep track of the execution outcomes
    execution_outcomes = {}
    # keep track of agents that have errors while parsing response to get actions
    get_action_from_response_errors = {}
    reward_tracker = {agent.agent_id: 0} 
    output_data_path = os.path.join(frame_folder, 'output_data.txt')
    response_data_path = os.path.join(frame_folder, 'response_data.json')
    global all_output_file
    all_output_file = os.path.join(frame_folder, 'all_output_data.txt')
    with open(all_output_file, 'w') as file:
        file.write(f"{agent.agent_type}, playing running with scissors arena \n\n")
    with open(output_data_path, 'a') as file:
        file.write(f"{agent.agent_type}, playing running with scissors arena \n\n")

    response_data_list = []
    user_message_data_list = []
    interaction_rewards = {}
    if agent.agent_type == 'hm' or agent.agent_type == 'planreact':
        hls_response, subgoal_response, hls_user_msg, subgoal_user_msg = await agent.two_level_plan(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)   
        print_and_save(f"Initial response: {hls_response}")
        print_and_save('\n')
        response_data_list.append(hls_response)
        user_message_data_list.append(hls_user_msg)
        # TO DO WHILE LOOP FOR VALID PLAN IN CLASS
    elif agent.agent_type == 'reflexion':
        state_info_dict = agent.get_state_info(state, step)
        goal_and_plan = {}
        state_info_dict_list = [state_info_dict]
        reward_during_plan = 0.0
        subgoal_user_msg, subgoal_response, goal_and_plan = await agent.subgoal_module(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step, goal_and_plan, state_info_dict_list, reward_during_plan)
    else:
        subgoal_user_msg, subgoal_response, goal_and_plan = await agent.subgoal_module(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
    goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
    print_and_save(agent.agent_id, goal_and_plan)
    response_data_list.append(subgoal_response)
    user_message_data_list.append(subgoal_user_msg)
    # set which subgoal we are on
    goal_and_plan['subgoal_num'] = 0
    agent_goals_and_actions[agent.agent_id] = goal_and_plan
    # combine all known states
    agent.combine_all_known_states(state)
    plan_grid = make_plan_grid(goal_and_plan, env, agent)
    agent.get_actions_from_plan(goal_and_plan, plan_grid, state)

    total_cost = agent.controller.total_inference_cost
    print_and_save(f"Step {step} Total Inference Cost:", total_cost)
    done = False
    during_interaction = False
    during_respawn = False
    after_interaction = False
    inventory_per_step = []

    while not done:
        step += 1
        """Agent actions"""
        step_actions_dict = {}
        agents_no_actions = []    # keep track agents that are out of actions
        goal_and_plan = agent_goals_and_actions[agent.agent_id]
        action = agent.act()

        # check if subgoal is completed, if so plan actions for the next subgoal
        if not action and len(goal_and_plan['action_plan']) != (goal_and_plan['subgoal_num']+1):
            # TO DO, if last subgoal was to pick up inventory, check that inventory was changed
            # May be hard to know if subgoal was intentionally to pick up inventory based on partially observable space
            goal_and_plan['subgoal_num'] += 1
            agent_goals_and_actions[agent.agent_id] = goal_and_plan
            # combine all known states
            agent.combine_all_known_states(state)
            plan_grid = make_plan_grid(goal_and_plan, env, agent)
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
                print_and_save(f"HERE IN THE INTERACTION LOOP")
        elif action:
            # use one step lookahead to see if action takes us to valid/intended location
            action, agent_goals_and_actions = agent.check_plan_one_step(action, state, env, agent_goals_and_actions)
            step_actions_dict[agent.agent_id] = ACTION_IDX[action]
        else:
            agents_no_actions.append(agent)

        # API call for agents that are out of actions
        if len(agents_no_actions) > 0 and not during_interaction:
            if agent.agent_type == 'reflexion':
                state_info_dict = agent.get_state_info(state, step)
                state_info_dict_list.append(state_info_dict)
                user_message, subgoal_response, goal_and_plan = await agent.subgoal_module(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step, goal_and_plan, state_info_dict_list, reward_during_plan)
                reward_during_plan = 0
            else:
                user_message, subgoal_response, goal_and_plan = await agent.subgoal_module(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
            print_and_save(f"User message: {user_message}")
            print_and_save(f"Response: {subgoal_response}")   
            print_and_save(agent.agent_id, goal_and_plan)
            user_message_data_list.append(user_message)
            response_data_list.append(subgoal_response)
            agent_goals_and_actions[agent.agent_id] = goal_and_plan                        
            parse_outcome = agent.get_actions_from_plan(goal_and_plan, plan_grid, state)
            get_action_from_response_errors[agent.agent_id] = parse_outcome
            action = agent.act()
            if action is None:
                print_and_save(f"Agent {agent.agent_id} is None, choosing NOOP.")
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
                action, agent_goals_and_actions = agent.check_plan_one_step(action, state, env, agent_goals_and_actions)
                step_actions_dict[agent.agent_id] = ACTION_IDX[action]
            #breakpoint()
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
        total_cost = agent.controller.total_inference_cost
        print_and_save(f"Step {step} Total Inference Cost:", total_cost, new_line=False)
        # update memory with currently observed states
        agent.update_memory(state, step)

        """Environment step"""
        if during_interaction:
            # if no player 0, take noop steps until environment is reset
            step_actions_dict = {'player_0': 0}
        state, obs, rewards, done, _, _ = env.step(step_actions_dict)
        save_obs_frames(obs, step, frame_folder)
        current_inventory = state['player_0']['inventory']
        inventory_per_step.append(current_inventory)
        # determine if we are in an interaction
        for key in state['global'].keys():
            if 'player_0_inter' in key and not during_interaction:
                during_interaction = True
                inter_id = agent.detect_inter_agent(state)
                current_inventory = state['player_0']['inventory']
                base_inventory = np.array([1., 1., 1.])
                if np.array_equal(base_inventory, current_inventory):
                    # weird bug where inventory is reset to base inventory
                    # Iterate backwards through inventory_per_step to find the most recent different inventory
                    for previous_inventory in reversed(inventory_per_step):
                        if not np.array_equal(base_inventory, previous_inventory):
                            current_inventory = previous_inventory
                            break  # Exit the loop once the most recent different inventory is found
                    print_and_save("Weird bug where inventory is reset to base inventory")
                    print_and_save(f"Inventory reset to base inventory. Previous inventory: {current_inventory}")
                interaction_inventory = {
                    'rock/yellow': int(current_inventory[0]), 
                    'paper/purple': int(current_inventory[1]), 
                    'scissors/blue': int(current_inventory[2])
                }
                break
        step_rewards = {agent.agent_id: 0}
        for agent_id, reward in rewards.items():
            # Round reward to 3 decimal points
            rounded_reward = np.round(reward, 3)
            reward_tracker[agent_id] += rounded_reward
            step_rewards[agent_id] += rounded_reward
            if agent.agent_type == 'reflexion':
                reward_during_plan += rounded_reward

        # determine if we are in respawning phase
        reset_env = any(key.startswith('player_0') for key in state['global'])
        if during_interaction and not reset_env and not during_respawn:
            print_and_save(f"Interaction: Agent {agent_id} with {inter_id}, inventory: {interaction_inventory}, received reward {reward}")
            after_interaction = True 
            interaction_rewards = {agent_id: np.round(reward, 3) for agent_id, reward in rewards.items()}
            during_respawn = True

        done = done['__all__']
        print_and_save(f"Step {step} rewards: {reward_tracker}, Player 0 inventory: {state['player_0']['inventory']}")
        if after_interaction and reset_env:
            during_interaction = False
            during_respawn = False
            agent.interaction_num += 1
            interaction_dict = {
                "Interaction": agent.interaction_num,
                "your_inventory": interaction_inventory,
                "rewards": interaction_rewards[agent.agent_id].item()
            }
            agent.interaction_history[inter_id].append(interaction_dict)
            print_and_save(f"Interaction {agent.interaction_num}")
            print_and_save(f"Step: {step}, Total reward: {reward_tracker}")
            print_and_save(f"Interaction {agent.interaction_num} with {inter_id}: inventory={interaction_inventory}, rewards={interaction_rewards[agent.agent_id].item()}")
            print_and_save('\n')
            with open(output_data_path, 'a') as file:
                file.write(f"Interaction {agent.interaction_num}, inventory={interaction_inventory}, rewards={interaction_rewards[agent.agent_id].item()}\n")
                file.write(f"{agent.interaction_history[inter_id]}\n")
                file.write(f"Step: {step}, Total reward: {reward_tracker}\n\n")
            # give feedback and reset all plans
            if agent.agent_type == 'hm' or agent.agent_type == 'planreact':
                hls_response, subgoal_response, hls_user_msg, subgoal_user_msg = await agent.two_level_plan(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step, after_interaction=True)
                if agent.agent_type == 'hm':
                    print_and_save(f"Opponent hypotheses: {agent.possible_opponent_strategy}")
                print_and_save(f"HLS: {hls_response}")
                print_and_save('\n')
                with open(output_data_path, 'a') as file:
                    file.write(f"HLS: {hls_response}\n\n")
                response_data_list.append(hls_response)
            elif agent.agent_type == 'reflexion':
                state_info_dict = agent.get_state_info(state, step)
                state_info_dict_list.append(state_info_dict)
                user_message, subgoal_response, goal_and_plan = await agent.subgoal_module(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step, goal_and_plan, state_info_dict_list, reward_during_plan, after_interaction=True)
                reward_during_plan = 0
            else:
                user_message, subgoal_response, goal_and_plan = await agent.subgoal_module(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step, after_interaction=True)
            print_and_save(f"Next subgoals: {subgoal_response}")
            with open(output_data_path, 'a') as file:
                file.write(f"Next subgoals: {subgoal_response}\n\n")
            response_data_list.append(subgoal_response)
            # Empty the action queue
            while not agent.all_actions.empty():
                agent.all_actions.get()
            goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
            print_and_save(agent.agent_id, goal_and_plan)
            # set which subgoal we are on
            goal_and_plan['subgoal_num'] = 0               
            agent_goals_and_actions[agent.agent_id] = goal_and_plan    
            # combine all known states
            agent.combine_all_known_states(state)
            plan_grid = make_plan_grid(goal_and_plan, env, agent)                   
            parse_outcome = agent.get_actions_from_plan(goal_and_plan, env.grid, state)
            get_action_from_response_errors[agent.agent_id] = parse_outcome
            after_interaction = False
            #breakpoint()
        # keep track of the execution outcomes
        if not during_interaction:
            execution_outcomes = {}
            execution_outcomes[agent.agent_id] =  agent.update_state(state)
    print_and_save(f"Episode finished at step {step} with rewards {reward_tracker}")

    # save results in minutes
    total_duration = time.time() - start_time
    total_duration = total_duration / 60
    # make dataframe - columns for agent_type, scenario, reward, datetime
    df_results = pd.DataFrame({'agent_type': [agent_label], 'scenario': [env.eval_num], 
                               'reward': [reward_tracker['player_0']], 'steps': [step], 'interaction_num': [agent.interaction_num],
                               'duration': [total_duration], 'cost': [total_cost], 'datetime': [date_time_str]})
    # df_results = pd.DataFrame({'agent_type': [full_agent_type], 'scenario': [scenario_num], 
    #                             'reward': [reward_tracker['player_0']], 'steps': [step], 
    #                             'avg_optimal': [avg_optimal], 'avg_actual': [avg_actual], 'difference': [diff],
    #                             'interaction_num': [agent.interaction_num], 'reward_per_interaction': [reward_tracker['player_0']/agent.interaction_num],
    #                             'duration': [total_duration], 'cost': [total_cost], 'datetime': [date_time_str]})
    all_results_file = './results/rws_arena_scores.csv'
    if os.path.exists(all_results_file):
        df_all_results = pd.read_csv(all_results_file, index_col=0)
        df_all_results = pd.concat([df_all_results, df_results], ignore_index=True)
    else:
        df_all_results = df_results
    df_all_results.to_csv(all_results_file)

    return frame_folder

        
