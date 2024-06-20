import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import re
import json
import asyncio
import random
import datetime
from PIL import Image
from meltingpot import substrate
from typing import Dict, List, Tuple, Optional, Any

from llm_plan.env.mp_llm_env import MeltingPotLLMEnv
from llm_plan.controller.async_llm import AsyncChatLLM
from llm_plan.agent.async_agent import DecentralizedAgent
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


def setup_agent(api_key, model_id, model_settings):
    llm = AsyncChatLLM(api_key=api_key)
    controller = AsyncGPTController(
        llm=llm,
        model_id=model_id,
        **model_settings
    )
    agent_config = {'agent_id': model_id}
    agent = DecentralizedAgent(agent_config, controller)    
    return agent


async def main_async():
    manual = False
    """Environment setup""" 
    substrate_name = 'running_with_scissors_in_the_matrix__repeated' 
    sprite_label_path = f'/ccn2/u/locross/llmv_marl/llm_plan/sprite_labels/{substrate_name}'
    env = MeltingPotLLMEnv(substrate_name, sprite_label_path)
    """Agent setup"""
    api_key_path = '/ccn2/u/locross/llmv_marl/llm_plan/lc_api_key.json'
    OPENAI_KEYS = json.load(open(api_key_path, 'r'))
    api_key = OPENAI_KEYS['API_KEY']
    model_settings = {
        "model": "gpt-4-1106-preview",
        "max_tokens": 2000,
        "temperature": 0.0,
        "top_p": 0.9,
        "n": 1,
    }
    agents = [
        setup_agent(api_key, model_id=f"player_{i}", model_settings=model_settings) 
        for i in range(2)
        ]
    """Interaction with the environment"""
    # initial step    
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
    # get prompts for each agent
    system_messages = [agent.system_message for agent in agents]
    user_messages = [agent.generate_initial_user_message(state, step) for agent in agents]        
    # # Gathering responses asynchronously
    # responses = await asyncio.gather(
    #     *[agent.controller.async_batch_prompt(system_msg, [user_msg]) for agent, system_msg, user_msg in zip(agents, system_messages, user_messages)]
    # )
    #breakpoint()
    responses = []
    agent_goals_and_actions = {}    
    for idx, agent in enumerate(agents):
        # response = responses[idx][0]        
        # goal_and_plan = agent.extract_goals_and_actions(response)
        # valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
        # counter = 0
        # while not valid_plan and counter < 4:
        #     print(f"Invalid plan for {agent.agent_id}. Trying again.")
        #     user_message = agent.generate_initial_user_message(state, step)
        #     plan_response = plan_response + user_message
        #     responses[idx] = await asyncio.gather(
        #         *[agent.controller.async_batch_prompt(system_messages[idx], [plan_response])]
        #     )
        #     response = responses[idx][0]
        #     #breakpoint()
        #     goal_and_plan = agent.extract_goals_and_actions(response)
        #     valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
        #     counter += 1
        # TEMP HARD CODED PLAN
        ego_state = state[agent.agent_id]
        player_position = [v for k, v in ego_state.items() if k.startswith(agent.agent_id)]
        if idx == 0:
            # goal_and_plan = {'goal': 'Collect blue resource', \
            #                 'action_plans': ['move_to('+str(player_position[0][0])+', (7, 11))','move_to((7, 11), (9, 11))','move_to((9, 11), (13, 3))']}
            goal_and_plan = {'goal': 'Collect blue resource', \
                            'action_plans': ['move_to('+str(player_position[0][0])+', (7, 7))','move_to((7, 7), (19, 7))']}
        elif idx == 1:
            # goal_and_plan = {'goal': 'Seek out and duel with other player', \
            #                 'action_plans': ['move_to('+str(player_position[0][0])+', (15, 5))','move_to((15, 5), (11, 7))','fire_at((11, 7))']}
            goal_and_plan = {'goal': 'Collect blue resource', \
                            'action_plans': ['move_to('+str(player_position[0][0])+', (19, 7))','move_to((19, 7), (7, 7))']}

        print(agent.agent_id, goal_and_plan)
        # set which subgoal we are on
        goal_and_plan['subgoal_num'] = 0
        agent_goals_and_actions[agent.agent_id] = goal_and_plan
        # make pathfinding grid include all resources excluding ones on the plan
        # Extracting coordinates from the action plans to exclude from grid
        waypoints = set()
        subgoal = goal_and_plan['action_plans'][goal_and_plan['subgoal_num']]
        tuples = re.findall(r'\((\d+,\s*\d+)\)', subgoal)
        for tup in tuples:
            waypoints.add(tuple(map(int, tup.split(','))))
        waypoints = list(waypoints)
        # combine all known states
        agent.combine_all_known_states(state) # update agent.all_known_states
        opponent_key = ['player_1' if agent.agent_id == 'player_0' else 'player_0'][0]
        labels = ['wall', 'yellow_box', 'blue_box', 'purple_box', opponent_key]
        plan_grid = env.build_grid_from_states(agent.all_known_states, labels, waypoints)
        agent.get_actions_from_plan(goal_and_plan, plan_grid, state)
        responses.append([goal_and_plan])

    total_cost = sum(agent.controller.total_inference_cost for agent in agents)
    print(f"Step {step} Total Inference Cost:", total_cost)
    breakpoint()
    done = False
    interaction = False
    # track rewards and responses
    reward_tracker = {agent.agent_id: 0 for agent in agents}
    response_data_list = [responses]
    response_data_path = os.path.join(frame_folder, 'response_data.json')
    user_message_data_list = [user_messages]
    system_messages_data_list = [system_messages] 

    #while not done:
    while step < 100:
        step += 1
        """Agent actions"""
        step_actions_dict = {}
        agents_no_actions = []    # keep track agents that are out of actions
        # keep track of the execution outcomes
        execution_outcomes = {}
        # keep track of agents that have errors while parsing response to get actions
        get_action_from_response_errors = {}
        for agent in agents:
            goal_and_plan = agent_goals_and_actions[agent.agent_id]
            action = agent.act()

            # check if subgoal is completed, if so plan actions for the next subgoal
            if not action and len(goal_and_plan['action_plans']) != (goal_and_plan['subgoal_num']+1):
                goal_and_plan['subgoal_num'] += 1
                agent_goals_and_actions[agent.agent_id] = goal_and_plan
                # Extracting coordinates from the action plans to exclude from grid
                waypoints = set()
                subgoal = goal_and_plan['action_plans'][goal_and_plan['subgoal_num']]
                tuples = re.findall(r'\((\d+,\s*\d+)\)', subgoal)
                for tup in tuples:
                    waypoints.add(tuple(map(int, tup.split(','))))
                waypoints = list(waypoints)
                # combine all known states
                agent.combine_all_known_states(state) # update agent.all_known_states
                labels = ['wall', 'yellow_box', 'blue_box', 'purple_box']
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
                # if not fire action to start interaction, add action back to buffer
                #if interact_action != 'FIRE_ZAP' and agent.all_actions.qsize() == 0:
                if not interaction and agent.all_actions.qsize() == 0:
                    # put INTERACT action back in to come back here on the next step
                    agent.all_actions.put(next_action)
                else:
                    print(f"HERE IN THE INTERACTION LOOP")
                #breakpoint()
            elif action:
                # use one step lookahead to see if action takes us to valid/intended location
                next_state_type, new_pos = agent.check_next_state_type(state, action)
                if next_state_type != 'ground' and new_pos != agent.destination and action != 'FIRE_ZAP' and subgoal[:7] != 'fire_at':
                    print(next_state_type)
                    # if next state is not ground, ie. collects unintended resource, replan with newly observed state information
                    # update current subgoal with current position
                    subgoal = goal_and_plan['action_plans'][goal_and_plan['subgoal_num']]
                    # Splitting the subgoal into two parts at the first closing parenthesis
                    part1, part2 = subgoal.split('),', 1)
                    # Updating the first part with the agent's current position
                    updated_part1 = part1[:part1.find('(') + 1] + str(agent.current_pos)
                    # Reassembling the updated subgoal
                    subgoal = updated_part1 + ',' + part2
                    goal_and_plan['action_plans'][goal_and_plan['subgoal_num']] = subgoal
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
                    breakpoint()
        
                step_actions_dict[agent.agent_id] = ACTION_IDX[action]
            else:
                agents_no_actions.append(agent)
            # API call for agents that are out of actions
            if len(agents_no_actions) > 0:
                breakpoint()
                system_messages = [agent.system_message for agent in agents_no_actions]
                user_messages = [
                    agent.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step) 
                    for agent in agents_no_actions]
                user_message_data_list.append(user_messages)
                system_messages_data_list.append(system_messages)
                # Gathering responses asynchronously
                responses = await asyncio.gather(
                    *[agent.controller.async_batch_prompt(system_msg, [user_msg]) 
                    for agent, system_msg, user_msg in zip(agents_no_actions, system_messages, user_messages)]
                )
                #breakpoint()
                response_data_list.append(responses)
            for idx, agent in enumerate(agents_no_actions):
                response = responses[idx][0]        
                goal_and_plan = agent.extract_goals_and_actions(response)
                # check that this is a valid plan
                valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
                counter = 0
                while not valid_plan and counter < 4:
                    print(f"Invalid plan for {agent.agent_id}. Trying again.")
                    user_message = agent.generate_initial_user_message(state, step)
                    plan_response = plan_response + user_message
                    responses[idx] = await asyncio.gather(
                        *[agent.controller.async_batch_prompt(system_messages[idx], [plan_response])]
                    )
                    response = responses[idx][0]
                    #breakpoint()
                    goal_and_plan = agent.extract_goals_and_actions(response)
                    if len(goal_and_plan) == 0:
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
                action = agent.act()
                if action is None:
                    print(f"Agent {agent.agent_id} is None, choosing NOOP.")
                    step_actions_dict[agent.agent_id] = 0
                else:
                    step_actions_dict[agent.agent_id] = ACTION_IDX[action]
            # reset agents_no_actions
            agents_no_actions = []
            # save prompt and responses
            with open(response_data_path, 'w') as file:
                data = {
                    'user_messages': user_message_data_list,
                    'system_messages': system_messages_data_list,
                    'response_data': response_data_list, 
                    'reward_tracker': reward_tracker
                    }
                json.dump(data, file, indent=4)
        total_cost = sum(agent.controller.total_inference_cost for agent in agents)
        print(f"Step {step} Total Inference Cost:", total_cost)
        # update memory with currently observed states
        for agent in agents:
            agent.update_memory(state, step)

        """Environment step"""
        #breakpoint()
        state, obs, rewards, done, _, _ = env.step(step_actions_dict)
        step_rewards = {agent.agent_id: 0 for agent in agents}
        for agent_id, reward in rewards.items():
            reward_tracker[agent_id] += reward
            step_rewards[agent_id] += reward
            if reward != 0:
                print(f"Interaction: Agent {agent_id} received reward {reward}")
                interaction = True 
            else:
                interaction = False
        done = done['__all__']
        save_obs_frames(obs, step, frame_folder)
        print(f"Step {step} rewards: {step_rewards}")
        if interaction:
            interaction_rewards = rewards
            # take noop steps until environment is reset
            reset_env = any(key.startswith('player_0') for key in state['global'])
            while not reset_env:
                # if no player 0 or player 1 in global state, then env is not reset
                step_actions_dict = {'player_0': 0, 'player_1': 0}
                state, obs, rewards, done, _, _ = env.step(step_actions_dict)  
                save_obs_frames(obs, step, frame_folder)
                reset_env = any(key.startswith('player_0') for key in state['global'])
                step += 1
            # give feedback and reset all plans
            # get prompts for each agent
            system_messages = [agent.system_message for agent in agents]
            user_messages = [
                    agent.generate_interaction_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, interaction_rewards, step) 
                    for agent in agents]
            user_message_data_list.append(user_messages)
            system_messages_data_list.append(system_messages)
            # Gathering responses asynchronously
            responses = await asyncio.gather(
                *[agent.controller.async_batch_prompt(system_msg, [user_msg]) 
                for agent, system_msg, user_msg in zip(agents, system_messages, user_messages)]
            )
            response_data_list.append(responses)
            for idx, agent in enumerate(agents):
                # Empty the action queue
                while not agent.all_actions.empty():
                    agent.all_actions.get()
                response = responses[idx][0]        
                goal_and_plan = agent.extract_goals_and_actions(response)
                # check that this is a valid plan
                valid_plan, plan_response = agent.is_valid_plan(state, goal_and_plan)
                counter = 0
                while not valid_plan and counter < 4:
                    print(f"Invalid plan for {agent.agent_id}. Trying again.")
                    user_message = agent.generate_initial_user_message(state, step)
                    plan_response = plan_response + user_message
                    responses[idx] = await asyncio.gather(
                        *[agent.controller.async_batch_prompt(system_messages[idx], [plan_response])]
                    )
                    response = responses[idx][0]
                    #breakpoint()
                    goal_and_plan = agent.extract_goals_and_actions(response)
                    if len(goal_and_plan) == 0:
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
        # keep track of the execution outcomes
        execution_outcomes = {}
        for agent in agents:
            execution_outcomes[agent.agent_id] =  agent.update_state(state)


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_async())


if __name__ == "__main__":
    main()