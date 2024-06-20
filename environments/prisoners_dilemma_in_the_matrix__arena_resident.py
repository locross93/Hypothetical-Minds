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

def print_and_save(agent_id, *args, new_line=True, **kwargs):
    global all_output_files
    print(*args, **kwargs)
    with open(all_output_files[agent_id], 'a') as file:
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
    labels = ['wall', 'green_box', 'red_box']
    for opponent_num in range(1, 8):
        opponent_key = f'player_{opponent_num}'
        labels.append(opponent_key)
    plan_grid = env.build_grid_from_states(agent.all_known_states, labels, waypoints)

    return plan_grid

def save_interaction_data(interaction_num, interaction_rewards, interaction_inventories, agent_ids, frame_folder):
    interaction_data = {
        'interaction_num': [interaction_num],
        f'{agent_ids[0]}_inventory': [interaction_inventories.get(agent_ids[0], 'nan')],
        f'{agent_ids[1]}_inventory': [interaction_inventories.get(agent_ids[1], 'nan')],
        f'{agent_ids[0]}_reward': [interaction_rewards[agent_ids[0]]],
        f'{agent_ids[1]}_reward': [interaction_rewards[agent_ids[1]]],
        f'{agent_ids[0]}_id': [agent_ids[0]],
        f'{agent_ids[1]}_id': [agent_ids[1]]
    }

    df_interaction = pd.DataFrame(interaction_data)
    interaction_file = os.path.join(frame_folder, f'interaction_{interaction_num}.csv')
    df_interaction.to_csv(interaction_file, index=False)

async def run_episode(env, agents):
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Create the subfolder path
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
    for agent in agents:
        agent.update_state(state)
    agent_goals_and_actions = {}    
    # Generate subgoals to execute given high-level strategy
    # keep track of the execution outcomes
    execution_outcomes = {}
    # keep track of agents that have errors while parsing response to get actions
    get_action_from_response_errors = {}
    reward_tracker = {agent.agent_id: 0 for agent in agents}
    output_data_path = os.path.join(frame_folder, 'output_data.txt')
    response_data_path = os.path.join(frame_folder, 'response_data.json')
    global all_output_files
    all_output_files = {}
    for agent in agents:
        all_output_files[agent.agent_id] = os.path.join(frame_folder, f'all_output_data_{agent.agent_id}.txt')
        with open(all_output_files[agent.agent_id], 'w') as file:
            file.write(f"Agent {agent.agent_id} playing prisoners dilemmma repeated \n\n")
    with open(output_data_path, 'a') as file:
        file.write(f"Playing prisoners dilemma repeated \n\n")

    response_data_list = []
    user_message_data_list = []
    interaction_rewards = {}

    # get prompts for each agent
    system_messages = [agent.system_message for agent in agents]
    user_messages = [agent.generate_initial_user_message(state) for agent in agents]        
    
    # Gathering responses asynchronously
    responses = await asyncio.gather(
        *[agent.controller.async_batch_prompt(system_msg, [user_msg]) for agent, system_msg, user_msg in zip(agents, system_messages, user_messages)]
    )

    for idx, agent in enumerate(agents):
        response = responses[idx][0]        
        goal_and_plan = agent.extract_goals_and_actions(response)
        print_and_save(agent.agent_id, agent.agent_id, goal_and_plan)
        response_data_list.append(response)
        user_message_data_list.append(user_messages[idx])
        # set which subgoal we are on
        goal_and_plan['subgoal_num'] = 0
        agent_goals_and_actions[agent.agent_id] = goal_and_plan
        # combine all known states
        agent.combine_all_known_states(state)
        plan_grid = make_plan_grid(goal_and_plan, env, agent)
        agent.get_actions_from_plan(goal_and_plan, plan_grid, state)

    total_cost = sum(agent.controller.total_inference_cost for agent in agents)
    print_and_save(agent.agent_id, f"Step {step} Total Inference Cost:", total_cost)
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
        for agent in agents:
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
                    print_and_save(agent.agent_id, f"HERE IN THE INTERACTION LOOP")
            elif action:
                # use one step lookahead to see if action takes us to valid/intended location
                action, agent_goals_and_actions = agent.check_plan_one_step(action, state, env, agent_goals_and_actions)
                step_actions_dict[agent.agent_id] = ACTION_IDX[action]
            else:
                agents_no_actions.append(agent)

        # API call for agents that are out of actions
        if len(agents_no_actions) > 0 and not during_interaction:
            system_messages = [agent.system_message for agent in agents_no_actions]
            user_messages = [
                agent.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
                for agent in agents_no_actions
            ]
            # Gathering responses asynchronously
            responses = await asyncio.gather(
                *[agent.controller.async_batch_prompt(system_msg, [user_msg])
                   for agent, system_msg, user_msg in zip(agents_no_actions, system_messages, user_messages)]
            )
            for idx, agent in enumerate(agents_no_actions):
                subgoal_response = responses[idx][0]
                print_and_save(agent.agent_id, f"User message: {user_messages[idx]}")
                print_and_save(agent.agent_id,  f"Response: {subgoal_response}")
                user_message_data_list.append(user_messages[idx])
                response_data_list.append(subgoal_response)
                goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
                print_and_save(agent.agent_id, agent.agent_id, goal_and_plan)
                agent_goals_and_actions[agent.agent_id] = goal_and_plan
                parse_outcome = agent.get_actions_from_plan(goal_and_plan, plan_grid, state)
                get_action_from_response_errors[agent.agent_id] = parse_outcome
                action = agent.act()
                if action is None:
                    print_and_save(agent.agent_id, f"Agent {agent.agent_id} is None, choosing NOOP.")
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
        print_and_save(agent.agent_id, f"Step {step} Total Inference Cost:", total_cost, new_line=False)
        # update memory with currently observed states
        for agent in agents:
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
            if 'player_0_inter' in key and not during_interaction:
                during_interaction = True
                inter_id = agent.detect_inter_agent(state)
                current_inventory = state['player_0']['inventory']
                base_inventory = np.array([1., 1.])

                if np.array_equal(base_inventory, current_inventory):
                    # weird bug where inventory is reset to base inventory
                    # Iterate backwards through inventory_per_step to find the most recent different inventory
                    for previous_inventory in reversed(inventory_per_step):
                        if not np.array_equal(base_inventory, previous_inventory):
                            current_inventory = previous_inventory
                            break  # Exit the loop once the most recent different inventory is found
                    print_and_save(agent.agent_id, "Weird bug where inventory is reset to base inventory")
                    print_and_save(agent.agent_id, f"Inventory reset to base inventory. Previous inventory: {current_inventory}")
                interaction_inventory = {
                    'cooperate/green': int(current_inventory[0]), 
                    'defect/red': int(current_inventory[1])
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
            print_and_save(agent.agent_id, f"Interaction: Agent {agent_id}, inventory: {interaction_inventory}, received reward {reward}")
            after_interaction = True 
            interaction_rewards = {agent_id: np.round(reward, 3) for agent_id, reward in rewards.items()}
            during_respawn = True

        done = done['__all__']
        save_obs_frames(obs, step, frame_folder)
        print_and_save(agent.agent_id, f"Step {step} rewards: {reward_tracker}, Player 0 inventory: {state['player_0']['inventory']}")
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
            print_and_save(agent.agent_id, f"Interaction {agent.interaction_num}")
            print_and_save(agent.agent_id, f"Step: {step}, Total reward: {reward_tracker}")
            print_and_save(agent.agent_id, f"Interaction {agent.interaction_num}: inventory={interaction_inventory}, rewards={interaction_rewards[agent.agent_id].item()}")
            print_and_save(agent.agent_id, '\n')
            # Save interaction data to CSV
            interaction_inventories = {agent.agent_id: interaction_inventory for agent in agents}
            save_interaction_data(agent.interaction_num, interaction_rewards, interaction_inventories, list(interaction_rewards.keys()), frame_folder)
            with open(output_data_path, 'a') as file:
                file.write(f"Interaction {agent.interaction_num}, inventory={interaction_inventory}, rewards={interaction_rewards[agent.agent_id].item()}\n")
                file.write(f"{agent.interaction_history}\n")
                file.write(f"Step: {step}, Total reward: {reward_tracker}\n\n")
            # give feedback and reset all plans
            for agent in agents:
                if agent.agent_type == 'hm':
                    hls_response, subgoal_response, hls_user_msg, subgoal_user_msg = await agent.two_level_plan(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step, after_interaction=True)
                    print_and_save(agent.agent_id, f"HLS: {hls_response}")
                    print_and_save(agent.agent_id, '\n')
                    with open(output_data_path, 'a') as file:
                        file.write(f"HLS: {hls_response}\n\n")
                    response_data_list.append(hls_response)
                elif agent.agent_type == 'reflexion':
                    state_info_dict = agent.get_state_info(state, step)
                    state_info_dict_list.append(state_info_dict)
                    user_message, subgoal_response, goal_and_plan = await agent.subgoal_module(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step, goal_and_plan, state_info_dict_list, reward_during_plan)
                    reward_during_plan = 0
                else:
                    user_message, subgoal_response, goal_and_plan = await agent.subgoal_module(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
                print_and_save(agent.agent_id, f"Next subgoals: {subgoal_response}")
                with open(output_data_path, 'a') as file:
                    file.write(f"Next subgoals: {subgoal_response}\n\n")
                response_data_list.append(subgoal_response)
                # Empty the action queue
                while not agent.all_actions.empty():
                    agent.all_actions.get()
                goal_and_plan = agent.extract_goals_and_actions(subgoal_response)
                print_and_save(agent.agent_id, agent.agent_id, goal_and_plan)
                # set which subgoal we are on
                goal_and_plan['subgoal_num'] = 0               
                agent_goals_and_actions[agent.agent_id] = goal_and_plan    
                # combine all known states
                agent.combine_all_known_states(state)
                plan_grid = make_plan_grid(goal_and_plan, env, agent)                                     
                parse_outcome = agent.get_actions_from_plan(goal_and_plan, plan_grid, state)
                get_action_from_response_errors[agent.agent_id] = parse_outcome
                after_interaction = False
        # keep track of the execution outcomes
        if not during_interaction:
            execution_outcomes = {}
            execution_outcomes[agent.agent_id] =  agent.update_state(state)
    print_and_save(agent.agent_id, f"Episode finished at step {step} with rewards {reward_tracker}")

    # save results in minutes
    total_duration = time.time() - start_time
    total_duration = total_duration / 60
    # make dataframe - columns for agent_type, scenario, reward, datetime, etc.
    df_results = pd.DataFrame({'agent_type': [agent_label for agent in agents],
                            'scenario': [env.eval_num] * len(agents),
                            'reward': [reward_tracker[agent.agent_id] for agent in agents],
                            'steps': [step] * len(agents),
                            'interaction_num': [agent.interaction_num] * len(agents),
                            'duration': [total_duration] * len(agents),
                            'cost': [agent.controller.total_inference_cost for agent in agents],
                            'datetime': [date_time_str] * len(agents)})
    all_results_file = './results/pd_arena_scores_resident.csv'
    if os.path.exists(all_results_file):
        df_all_results = pd.read_csv(all_results_file, index_col=0)
        df_all_results = pd.concat([df_all_results, df_results], ignore_index=True)
    else:
        df_all_results = df_results
    df_all_results.to_csv(all_results_file)

    return frame_folder

        


        
