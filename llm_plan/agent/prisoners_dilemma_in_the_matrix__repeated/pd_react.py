import re
import abc
import ast
import asyncio
import numpy as np
import math
from copy import deepcopy
from queue import Queue
from typing import List, Dict, Any, Tuple, Optional
from itertools import permutations

from llm_plan.agent import action_funcs

class DecentralizedAgent(abc.ABC):
    def __init__(
            self, 
            config: Dict[str, Any],
            controller: Any,
            ) -> None:
        self.agent_id = config['agent_id']
        self.config = config
        self.controller = controller        
        self.all_actions = Queue()
        self.generate_system_message()
        self.memory_states = {}
        self.interact_steps = 0
        self.interaction_history = []
        self.opponent_hypotheses = {}
        self.interaction_num = 0
        player_key = self.agent_id
        opponent_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
        for entity_type in ['green_box', 'red_box', 'ground', player_key, opponent_key]:
            self.memory_states[entity_type] = []


    def generate_system_message(self):
        self.system_message = f"""
            You are Agent {self.agent_id} in the two player 'prisoner_dilemma_in_the_matrix__repeated'
            Melting Pot multiagent reinforcement learning environment that is an 23x15 (x by y) grid with resources to collect and walls to navigate around. 
            Your goal is to maximize reward.
            Players can move around the map and collect resources of 2 discrete types corresponding to cooperate and
            defect strategies - Green box = cooperate  - Red box = defect. 
            You and another agent will collect resources and compare inventories in an encounter in repeated rounds. 
            This is a mixed motive game where consequences/rewards of the inventory comparison are congruent with the classic Prisoner’s Dilemma matrix game, and since you will interact repeatedly this is a version of iterated Prisoner’s Dilemma.           
            In addition to movement, the agents have an action to fire an "interaction" beam which initiates a duel to compare inventories and distribute rewards.
            All players carry an inventory with the count of resources picked up since last respawn and for each respawn start with an inventory of 1 resource each.
            This inventory is visible in the state with the key 'inventory'. 

            Here is how the payoff matrix in the generic prisoner dilemma works:
            - If both players cooperate, they both receive a reward of around 3,
            - If one player cooperates and the other defects, the defector receives a reward of around 5 and the cooperator receives a reward of around 0,
            - If both players defect, they both receive a reward of around 1.

            The nuance of 'prisoner_dilemma_in_the_matrix__repeated' is that the rewards are using the payoff rules of the classic prisoner dilemma, but the interaction is repeated and the rewards are distributed based on the inventory of resources collected by each player.
            The rules for the rewards based on the inventory are as follows:
            - If both agent cooperate, the one with the more cooperate resources will receive a reward lower than the one with less cooperate resources,
            - If one agent cooperates and the other defects, the more defect resources the defector has, the higher the reward for the defector,
            - If both agents defect, the one with more defect resources will receive a higher reward than the one with less defect resources.            

            Your goal before each interaction is to try and infer what the other player will play and how their strategy over time is effected by your plays.
            Then pick a strategy accordingly to maximize reward.
            You will only want to pick up one type of resource before an interaction, to not waste time collecting other resources.
            Since the reward function is based on the ratio of the two inventories, you will only want to pick up one type of resource before an interaction.
            For example the inventories {{'cooperate/green': 1, 'defect/red': 1}} and {{'cooperate/green': 3, 'defect/red': 3}} will both result in the same reward, so don't waste time collecting more than you need.
            Player 1 will also always only pick up one type of resource before an interaction.
            To play a strategy strongly, pick up at least 6 resources or more of only one color and then fire the interaction beam at the other player.
            To commit less strongly to a strategy, pick up around 2 resources of only color and then fire the interaction beam at the other player.
            State Description: This environment is partially-observable, you can observed a 5x5 grid around your agent depending on your position and orientation (you can see more in front of you than behind).
            Previously seen states will be represented in memory, but note that these states could potentially be outdated. For example the other agent could collect a resource that you previously saw.
            Given the partially-observable nature of the environment, you will need to explore the environment appropriately and select goals based on the information you've gathered.
            Also pay attention to your Player 1's position when you see it in order to duel with them and gain information about their strategy.
            Your goal is to maximize reward attained over an entire episode, so keep in mind the long-term consequences of your actions. Look at events in a gestalt manner.
            """
    
    def generate_hls_user_message(self, state, step):
        ego_state = state[self.agent_id]
        # Extracting information from the state
        #map_size = "18x24"  # Assuming the map size is constant
        map_size = "23x15"  # Assuming the map size is constant
        player_position = {k: v for k, v in ego_state.items() if k.startswith(self.agent_id)}
        player_orientation = list(player_position.keys())[0].split('-')[-1]
        player_inventory = list(ego_state['inventory'])
        # get locations of non wall keys in state
        movable_locations = []
        for k, v in state['global'].items():
            if k != 'wall':
                for loc in v:
                    movable_locations.append(loc)
        # movable_locations = [v for k, v in state['global'].items() if k != 'wall']
        red_locations = ego_state.get('red_box', [])
        green_locations = ego_state.get('green_box', [])
        beam_locations = ego_state.get('beam', [])
        ground_locations = ego_state.get('ground', [])
        opponent_locations = [v[0] for k, v in ego_state.items() if k.startswith('player_1' if self.agent_id == 'player_0' else 'player_0')]

        strategy_request = f"""
            Strategy Request:
            You are at step {step} of the game.
            Provide a strategy for player {self.agent_id}. 
            Your response should outline a high level strategy - which strategy do you want to take first and why?
            This response will be shown to you in the future in order for you to select lower-level actions to implement this strategy.
            Example response:
            High level strategy: I want to play a strong defect strategy and collect 5 red resources.
            You will be prompted again shortly to select subgoals and action plans to execute this strategy, so do not include that in your response yet right now.
            """
        user_message = f"""Current State Description:
            - Global Map Size: {map_size} grid (Walls are located at the boundaries of the map and in other places that are invalid for move_to).
            - Valid Locations for move_to: {movable_locations}
            - Player Position: {player_position}
            - Player Orientation: {player_orientation}
            - Player Inventory (green, red): {player_inventory}
            - Egocentric Observations Size: 5x5 grid around your agent. You currently can observe the following based on your position and orientation:
            - Observable Red Box Locations: {red_locations if 'red_box' in ego_state else []}
            - Observable Green Box Locations: {green_locations if 'green_box' in ego_state else []}
            - Observable Opponent Locations: {opponent_locations}
            - Previously seen states from memory (format: ((x,y), step last observed, distance from current location): {self.memory_states}
            
            {strategy_request}
            """
        user_message = f"For agent {self.agent_id}: {user_message} Provide a strategy only for this agent."
        return user_message
    
    def calculate_manhattan_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> int:
        """Calculate the Manhattan distance between two points."""
        if point1 is None or point2 is None:
            return float('inf')
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def update_memory_states_with_distance(self, current_location: Tuple[int, int]):
        """Update memory states with distance from the current location."""
        for entity_type, locations in self.memory_states.items():
            for i, (location, step_last_observed, distance) in enumerate(locations):
                distance = f"distance: {self.calculate_manhattan_distance(current_location, location)}"
                # Update the tuple with the distance information
                self.memory_states[entity_type][i] = (location, step_last_observed, distance)

    def generate_feedback_user_message(
            self, 
            state,
            execution_outcomes, 
            get_action_from_response_errors,
            rewards,
            step,
            after_interaction=False):
        ego_state = state[self.agent_id]
        # Extracting information from the state
        #map_size = "18x24"  # Assuming the map size is constant
        map_size = "23x15"  # Assuming the map size is constant
        player_position = {k: v for k, v in ego_state.items() if k.startswith(self.agent_id)}
        player_orientation = list(player_position.keys())[0].split('-')[-1]
        player_inventory = list(ego_state['inventory'])
        # get locations of non wall keys in state
        movable_locations = []
        for k, v in state['global'].items():
            if k != 'wall':
                for loc in v:
                    movable_locations.append(loc)

        player_position_list = next(iter(player_position.values()))
        current_position = player_position_list[0] if player_position_list else None
        if current_position is not None:
            self.update_memory_states_with_distance(player_position_list[0])

        #movable_locations = [v for k, v in state['global'].items() if k != 'wall']
        red_locations = ego_state.get('red_box', [])
        green_locations = ego_state.get('green_box', [])
        beam_locations = ego_state.get('beam', [])
        ground_locations = ego_state.get('ground', [])
        opponent_locations = [v[0] for k, v in ego_state.items() if k.startswith('player_1' if self.agent_id == 'player_0' else 'player_0')]

        # Calculate the distance from the current location to the red and green locations
        red_locations_with_distance = [(loc, f"distance: {self.calculate_manhattan_distance(current_position, loc)}") for loc in red_locations]
        green_locations_with_distance = [(loc, f"distance: {self.calculate_manhattan_distance(current_position, loc)}") for loc in green_locations]
        
        rewards_str = "\n".join(f"- {player}: {reward}" for player, reward in rewards.items())
        
        if after_interaction:
            strategy_preamble = f"""
            Strategy Request:
            An interaction with the other player has occurred at step {step}, {self.interaction_history[-1]}.
            """
        else:
            strategy_preamble = f"""
            Strategy Request:
            You are at step {step} of the game.
            """
        
        strategy_request = f"""
            Select subgoals in order to maximize rewards. 
            Think about what strategy your opponent is playing and select a strategy to counter it based on the interaction history: {self.interaction_history}. 
            Your task is to devise efficient action plans for player {self.agent_id}, reason through what the next subgoals should be given the state information. 
            Your response should be broken up into two parts:
                1. Subgoal Plan - based on the current state and the high-level strategy you previously specified above, 
                decompose this strategy into a sequence subgoals and actions to efficiently implement this strategy. Think step by step about this. This could be fairly long.
                2. Action Plan - output this sequence of actions in the following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
            Example response 1:
            Subgoal Plan: Given the current state, my cooperate strategy to get to an inventory of 1 red and 5 green, and my inventory of 1 red and 3 green resources,
            I should move towards the two nearest observable green box location to collect so I have 5.
            The nearest observable green boxes are at (9, 5) and (13, 5).\n- Since I am at (11, 7), the closest one is at (9, 5).
            I should move there first and then move to (13, 5). After these two actions are completed, I can move towards the middle of the environment to initiate a duel.
            ```python
            {{
              'action_plan': ['move_to((11, 7), (9, 5))', 'move_to((9, 5), (13, 5))']
            }}
            ```
            Example response 2:
            Subgoal Plan: Currently, my inventory is 1 green and 4 red, and I am at position (10, 10). 
            My strategy is to strengthen the defect strategy by collecting more red resources. The nearest observable red boxes are at (10, 9) and (8, 11). 
            I will first move to (10, 9) to collect the red resource, then move to (8, 11) to collect another red resource, aiming for an inventory of 1 green and 6 red. 
            After collecting these resources, I will look for the opponent to initiate a duel.
            ```python
            {{
            'action_plan': ['move_to((10, 10), (10, 9))', 'move_to((12, 9), (8, 11))']
            }}
            ```
            Example response 3:
            Subgoal Plan: I start with an inventory of 3 green and 1 red, positioned at (8, 5). My strategy is to adopt a cooperate strategy by collecting more green resources. 
            The nearest observable green boxes are at (7, 4) and (3, 6). I plan to move to (7, 4) first to pick up the green resource, then proceed to (3, 6) to collect another green resource, 
            targeting an inventory of 5 green and 1 red. Once I have collected these resources, I will attempt to find and duel with the opponent.
            ```python
            {{
            'action_plan': ['move_to((8, 5), (7, 4))', 'move_to((7, 4), (3, 6))']
            }}
            ```
            DO NOT PUT ANY COMMENTS INSIDE THE [] OF THE ACTION PLAN. ONLY PUT THE ACTION FUNCTIONS AND THEIR ARGUMENTS.
            The strategy should be efficient, considering the shortest paths to resources and strategic positioning for duels. 
            Format the dictionary as outlined below, listing the strategy and action plans.
            Do not use JSON or any other formatting. 
            Actions should align with the action functions, \
            emphasizing efficient pathfinding and playing the corresponding strategies.
            Consider the entire game state to plan the most efficient paths for resource collection and strategy execution.
            To do this you will need to think step by step about what actions to output in the following format for 
            these players to efficiently collect the appropriate resources/target inventories and play their strategy.
            Take into account the proximity of the target_coord from the src_coord and the shortest path to get to a target resource.
            
            ONLY USE THESE 2 ACTION FUNCTIONS:
            - move_to(src_coord, target_coord): Efficiently move agent from source coordinate to target coordinate. DO NOT move to any locations where walls are present.
            - fire_at(target_coord): Stay around specified coordinate and fire interaction when opponent is spotted to initiate duel. 
            After an interaction both players respawn.

            Use the fact that you also have distance given to you for a resource from your current location.
            
            Keep plans relatively short (<6 actions), especially at the early steps of an episode. You will be prompted again when the action plans are finished and more information is observed.
            """

        strategy_request = strategy_preamble + strategy_request

        user_message = f"""Current State Description:
            - Global Map Size: {map_size} grid (Walls are located at the boundaries of the map and in other places that are invalid for move_to).
            - Valid Locations for move_to: {movable_locations}
            - Player Position: {player_position}
            - Player Orientation: {player_orientation}
            - Player Inventory (green, red): {player_inventory}
            - Egocentric Observations Size: 5x5 grid around your agent. You currently can observe the following based on your position and orientation:
            - Observable Red Box Locations (format: ((x,y), distance from current location): {red_locations_with_distance}
            - Observable Green Box Locations (format: ((x,y), distance from current location): {green_locations_with_distance}
            - Observable Opponent Locations: {opponent_locations}
            - Previously seen states from memory (format: ((x,y), step last observed, distance from current location): {self.memory_states}

            Execution Outcomes:
            {execution_outcomes}

            Error for extracting and executing actions from the response:
            {get_action_from_response_errors}

            Rewards:
            {rewards_str}

            {strategy_request}
            """
            
        return user_message

    async def subgoal_module(self, state, execution_outcomes, get_action_from_response_errors, reward_tracker, step, after_interaction=False):
        user_message = self.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
        responses = await asyncio.gather(
            *[self.controller.async_batch_prompt(self.system_message, [user_message])]
        )
        subgoal_response = responses[0][0]  
        goal_and_plan = self.extract_goals_and_actions(subgoal_response)
        # check that this is a valid plan
        valid_plan, plan_response = self.is_valid_plan(state, goal_and_plan)
        counter = 0
        while not valid_plan and counter < 10:
            print(f"Invalid plan for {self.agent_id}, {plan_response}. Trying again.")
            user_message = self.generate_feedback_user_message(state, plan_response, get_action_from_response_errors, reward_tracker, step)
            plan_response = plan_response + user_message
            responses = await asyncio.gather(
                *[self.controller.async_batch_prompt(self.system_message, [plan_response])]
            )
            subgoal_response = responses[0][0]
            goal_and_plan = self.extract_goals_and_actions(subgoal_response)
            valid_plan, plan_response = self.is_valid_plan(state, goal_and_plan)
            counter += 1

        # set which subgoal we are on
        goal_and_plan['subgoal_num'] = 0 

        return user_message, subgoal_response, goal_and_plan

    def extract_dict(self, response):
        try:
            # Find the start of the dictionary by looking for the "player_0" key
            # We will use a pattern that is likely to be unique to the start of the dictionary
            start_marker = "```python\n"
            end_marker = "\n```"         
            start_index = response.find(start_marker) + len(start_marker)
            end_index = response.find(end_marker, start_index)
            if start_index == -1 or end_index == -1:
                raise ValueError("Python dictionary markers not found in GPT-4's response.")
            dict_str = response[start_index: end_index].strip()

            # Process each line, skipping lines that are comments
            lines = dict_str.split('\n')
            cleaned_lines = []
            for line in lines:
                # Check if line contains a comment
                comment_index = line.find('#')
                if comment_index != -1:
                    # Exclude the comment part
                    line = line[:comment_index].strip()
                if line:  # Add line if it's not empty
                    cleaned_lines.append(line)

            # Reassemble the cleaned string
            cleaned_dict_str = ' '.join(cleaned_lines)
         
            # Convert the string representation of a dictionary into an actual dictionary
            extracted_dict = ast.literal_eval(dict_str)
            return extracted_dict
        except Exception as e:
            print(f"Error parsing dictionary: {e}")
            return {}


    def extract_goals_and_actions(self, response):
        goals_and_actions = self.extract_dict(response)
        return goals_and_actions


    def get_actions_from_plan(
            self, 
            goals_and_actions: Dict[str, Any],
            grid: np.ndarray,
            state: Dict[str, Any],
            return_actions = False) -> Optional[str]:
        """Given a plan, return a list of actions to be performed by the agent."""
        actions_collected = []
        try:
            #self.goal = goals_and_actions['goal']
            self.action_plan = goals_and_actions['action_plan']
            self.subgoal_num = goals_and_actions['subgoal_num']
        except Exception as e:
            return f"Error parsing goals and actions: {e}" 
        #for action_plan in self.action_plans:
        # get actions for current subgoal
        action_plan = self.action_plan[self.subgoal_num]
        split_idx = action_plan.find('(')
        func_name = action_plan[:split_idx]
        # if error, return the error message to be appended to the prompt
        try:
            func_args = ast.literal_eval(action_plan[split_idx+1:-1])
        except Exception as e:
            return f"Error parsing function arguments: {e}"            
        func = getattr(action_funcs, func_name)            
        if func_name == 'move_to':                
            start, goal = func_args
            #optimal = find_optimal(self.action_plan)                  
            paths, actions, current_orient, path_found = func(start, goal, grid, self.orientation)
            #paths_optimal, actions_optimal, current_orient_optimal, path_optimal = func(start, optimal, grid, self.orientation)
            if not path_found:
                print(f"No path found for action plan: {action_plan}. Making less strict action sequence")
                self.combine_all_known_states(state) # update agent.all_known_states
                goal_type = None
                for key, coordinates in self.all_known_states.items():
                    if goal in coordinates:
                        goal_type = key
                opponent_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
                labels = ['wall', 'green_box', 'red_box', opponent_key]
                # remove goal_type from obstacles, ie can collect resources of same type
                if goal_type in labels:
                    labels.remove(goal_type)
                new_grid = self.build_grid_from_states(self.all_known_states, labels, [goal])
                paths, actions, current_orient, path_found = func(start, goal, new_grid, self.orientation)
            if not path_found:
                print(f"Still no path found for action plan: {action_plan}. Making even less strict action sequence")
                # only include walls as obstacles
                labels = ['wall']
                new_grid = self.build_grid_from_states(self.all_known_states, labels, [goal])
                paths, actions, current_orient, path_found = func(start, goal, new_grid, self.orientation)

            # update agent's position (if moved) and orientation
            if len(paths) > 0:
                self.pos = paths[-1]
            self.orientation = current_orient
            self.destination = goal
        elif func_name == 'fire_at':
            actions = ['INTERACT_'+str(func_args)]
            # target = func_args
            # actions, current_orient = func(self.pos, self.orientation, target)
            # self.orientation = current_orient

        for action in actions:                      
            if return_actions:
                actions_collected.append(action)  # Collect actions if return_actions is True
            else:
                self.all_actions.put(action)  # Queue actions for execution otherwise

        if return_actions:
            return actions_collected  # Return collected actions if requested

    def sort_resources_by_distance(self, current_position, resource_positions):
        total_distance = 0
        position = current_position
        for resource in resource_positions:
            if resource is not None:
                distance = abs(position[0] - resource[0]) + abs(position[1] - resource[1])
                total_distance += distance
                position = resource
        return total_distance
    
    def calculate_optimal_subgoals(self, current_position, resource_positions, N, grid, state):
        #sorted_resources = self.sort_resources_by_distance(current_position, resource_positions)
        resource_permutations = permutations(resource_positions, N)
        #breakpoint()
        min_distance = float('inf')
        optimal_sequence = None
        for perm in resource_permutations:
            distance = self.sort_resources_by_distance(current_position, perm)
            if distance < min_distance:
                min_distance = distance
                optimal_sequence = perm
        print(f"SORTED RESOURCES {optimal_sequence}")
        action_plans = []
        total_actions = 0
        if optimal_sequence == None or len(optimal_sequence) == 0:
            return {'action_plan': action_plans, 'total_actions': total_actions}
        for i in range(int(N)):
            target_resource = optimal_sequence[i]
            print(f"target resource: {target_resource}")
            func = getattr(action_funcs, 'move_to')
            print(f"CURRENT POSITION {current_position}") 
            paths, actions, current_orient, path_found = func(current_position, target_resource, grid, self.orientation)
            print(f"PATHS:{paths}")
            print(f"actions:{actions}")
            print(f"current orientation:{current_orient}")
            print(f"Path Found:{path_found}")

            if not path_found:
                print(f"No path found for action plan: {action_plans}. Making less strict action sequence")
                #self.combine_all_known_states(state) # update agent.all_known_states
                goal_type = None
                for key, coordinates in self.all_known_states.items():
                    if target_resource in coordinates:
                        goal_type = key
                opponent_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
                labels = ['wall', 'green_box', 'red_box', opponent_key]
                # remove goal_type from obstacles, ie can collect resources of same type
                if goal_type in labels:
                    labels.remove(goal_type)
                new_grid = self.build_grid_from_states(self.all_known_states, labels, [target_resource])
                paths, actions, current_orient, path_found = func(current_position, target_resource, new_grid, self.orientation)
            
            if not path_found:
                print(f"Still no path found for action plan: {action_plan}. Making even less strict action sequence")
                # only include walls as obstacles
                labels = ['wall']
                new_grid = self.build_grid_from_states(self.all_known_states, labels, [target_resource])
                paths, actions, current_orient, path_found = func(current_position, target_resource, new_grid, self.orientation)
            
            if len(paths) > 0:
                action_plans.append(f"move_to({current_position}, {target_resource})")
                current_position = paths[-1]  # Assuming paths contains the final position
                total_actions += len(actions)
                self.orientation = current_orient  # Update agent's orientation
                
        return {'action_plan': action_plans, 'total_actions': total_actions}


    def build_grid_from_states(self, states: Dict[str, List[Tuple[int, int]]], labels: List[str], ignore: List[Tuple[int, int]] = None) -> np.ndarray:
        """Build a grid from a dictionary of states. Setting walls and other obstacles you pass in (as str in labels) to 1, 
        and all other things as 0."""
        grid_width = 23
        grid_height = 15
        grid = np.zeros((grid_width, grid_height))
        for label, coords in states.items():
            if label == 'inventory' or coords is None:
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
    


    def is_valid_plan(self, state, goals_and_actions):
        """
        Check if the action plans contain any moves that lead to a wall.

        :return: Boolean indicating whether the plans are valid or not.
        """
        if 'action_plan' not in goals_and_actions or len(goals_and_actions['action_plan']) == 0:
            response = f"Error: No action plan found in response: {goals_and_actions}. Replan and try again."
            return False, response
            
        wall_locations = state['global']['wall']        
        for plan in goals_and_actions['action_plan']:
            # Extract the destination location from the action plan
            try:
                destination = tuple(map(int, plan.split('(')[-1].strip(')').split(', ')))
            except ValueError:
                response = f"Invalid destination location in action plan: {plan}. Replan and try again."
                return False, response
            
            # Check if the destination is a wall location
            if destination in wall_locations:
                response = f"Invalid plan as it leads to a wall: {destination}. Replan and try again. DO NOT INCLUDE MOVE_TO ACTIONS THAT LEAD TO WALLS. \
                Think step by step about whether target locations are valid. "
                return False, response  # Invalid plan as it leads to a wall
            
            if destination[0] < 0 or destination[0] > 22 or destination[1] < 0 or destination[1] > 14:
                response = f"Invalid plan as it leads to a wall: {destination}. Replan and try again. DO NOT INCLUDE MOVE_TO ACTIONS THAT LEAD TO WALLS. \
                Think step by step about whether target locations are valid. "
                return False, response

        response = None
        return True, response  # All plans are valid


    def act(self) -> Optional[str]:
        """Return the next action to be performed by the agent. 
        If no action is available, return None.
        """
        if not self.all_actions.empty():        
            return self.all_actions.get()

    def update_state(self, state: Dict[str, Any]) -> Optional[str]:
        """Update the position of the agent."""
        try:
            agent_key = [item for item in state['global'].keys() if item.startswith(self.agent_id)][0]        
            self.pos = state['global'][agent_key][0]    # (x, y) -> (col, row)                    
            self.orientation = agent_key.split('-')[1]            
            if hasattr(self, 'goal'):                                
                # checking if agent is at it's goal location
                for action_plan in self.action_plan:
                    if 'move_to' in action_plan:
                        split_idx = action_plan.find('(')
                        func_args = ast.literal_eval(action_plan[split_idx+1:-1])
                        goal_pos = func_args[1]                                
                        output = f"Reached goal position {goal_pos}: {self.pos == goal_pos}"
                        return output
        except IndexError:
            print(f"Agent {self.agent_id} error...")
            if hasattr(self, 'goal'):                                
                # checking if agent is at it's goal location
                for action_plan in self.action_plan:
                    if 'move_to' in action_plan:
                        split_idx = action_plan.find('(')
                        func_args = ast.literal_eval(action_plan[split_idx+1:-1])
                        goal_pos = func_args[1]                                
                        output = f"Out of game now due to hit by laser in the last 5 steps."
                        return output
    
    def find_optimal(self, change_in_inventory):
        """
        Finds closest location for a given type of a resource from the agent's memory.

        :param change_in_inventory: Denotes how many green/red resources needed ('green_box' or 'red_box').
        :return closest_location is location of nearest 'resource' type resource
                steps is what move_to returns for going there
        """
        
        steps_taken = 0
        green_needed = change_in_inventory[-1][0]
        red_needed = change_in_inventory[-1][1]

        for i in range(green_needed):
            ego_state = state[self.agent_id]
            current_location = [v[0] for k, v in ego_state.items() if k.startswith('player_1' if self.agent_id == 'player_1' else 'player_0')]
            resource_locations = self.memory_states['green']
            closest_location = None
            min_distance = float('inf')

            for location, _ in resource_locations:
                distance = sqrt((location[0] - current_location[0])**2 + (location[1] - current_location[1])**2)
                if distance < min_distance:
                    closest_location = location
                    min_distance = distance
            ##execute move_to action, update steps and update everything else including current location
            ##How to run move_to?

        for i in range(red_needed):
            ego_state = state[self.agent_id]
            current_location = [v[0] for k, v in ego_state.items() if k.startswith('player_1' if self.agent_id == 'player_1' else 'player_0')]
            resource_locations = self.memory_states['red']
            closest_location = None
            min_distance = float('inf')

            for location, _ in resource_locations:
                distance = sqrt((location[0] - current_location[0])**2 + (location[1] - current_location[1])**2)
                if distance < min_distance:
                    closest_location = location
                    min_distance = distance
            ##execute move_to action, update steps and update everything else including current location
            ##How to run move_to and get length of steps?
        return steps_taken

    def update_memory(self, state, step):
        """
        Update the memory states with the latest observations.

        :param state: Dictionary of the current state.
        :param step: The current step number in the game.
        """
        ego_state = state[self.agent_id]
        player_key = self.agent_id
        opponent_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
        for entity_type in ['green_box', 'red_box', 'ground', player_key, opponent_key]:
            if entity_type[:6] == 'player':
                observed_locations = [v[0] for k, v in ego_state.items() if k.startswith(entity_type)]
            else:
                observed_locations = ego_state.get(entity_type, [])
            for location in observed_locations:
                # Remove this location from all other entity types
                for other_entity in self.memory_states:
                    if other_entity != entity_type:
                        self.memory_states[other_entity] = [
                            (loc, step_str, distance) for loc, step_str, distance in self.memory_states[other_entity] if loc != location
                        ]

                # Update or add the location with the latest step number and remove older references of the same location
                #breakpoint()
                self.memory_states[entity_type] = [
                    (loc, step, distance) for loc, step, distance in self.memory_states[entity_type] if loc != location
                ]
                self.memory_states[entity_type].append((location, 'Step: '+str(step), 'distance: unknown'))


    def interact(self, state, location):
        """
        Interact with the target player.

        :param state: Dictionary of the current state.
        :param location: The location to stay around.
        :return: The action to be performed by the agent.
        """
        ego_state = state[self.agent_id]
        opponent_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
        opponent_locations = [v[0] for k, v in ego_state.items() if k.startswith(opponent_key)]
        if len(opponent_locations) > 0:
            target = opponent_locations[0]
            fire_at = getattr(action_funcs, 'fire_at') 
            actions, current_orient = fire_at(self.pos, self.orientation, target)
            for action in actions:                      
                self.all_actions.put(action)
        else:
            # spin clockwise until other agent is found
            self.all_actions.put('TURN_RIGHT')

    def check_next_state_type(self, state, action):
        # used to determine move outcome based on orientation
        action_outcome_dict = {
            'N': {'FORWARD': (0, -1), 'STEP_LEFT': (-1, 0), 'STEP_RIGHT': (1, 0), 'BACKWARD': (0, 1),
                'TURN_LEFT': 'W', 'TURN_RIGHT': 'E', 'FIRE_ZAP': 'N'},
            'E': {'FORWARD': (1, 0), 'STEP_LEFT': (0, -1), 'STEP_RIGHT': (0, 1), 'BACKWARD': (-1, 0),
                'TURN_LEFT': 'N', 'TURN_RIGHT': 'S', 'FIRE_ZAP': 'E'},
            'S': {'FORWARD': (0, 1), 'STEP_LEFT': (1, 0), 'STEP_RIGHT': (-1, 0), 'BACKWARD': (0, -1),
                'TURN_LEFT': 'E', 'TURN_RIGHT': 'W', 'FIRE_ZAP': 'S'},
            'W': {'FORWARD': (-1, 0), 'STEP_LEFT': (0, 1), 'STEP_RIGHT': (0, -1), 'BACKWARD': (1, 0),
                'TURN_LEFT': 'S', 'TURN_RIGHT': 'N', 'FIRE_ZAP': 'W'},
            } 
        # Extracting the current position and orientation of the player
        ego_state = state[self.agent_id]
        for key, value in ego_state.items():
            if self.agent_id in key:
                self.current_pos = value[0]
                self.current_orient = key.split('-')[-1]
                break
        else:
            # Return if player position or orientation is not found
            return "Player position or orientation not found", None

        # Determining the movement based on the action and orientation
        movement = action_outcome_dict[self.current_orient][action]
        if not isinstance(movement, tuple):
            # movement = (0,0) if movement is not a tuple (i.e., it's a turn action)
            movement = (0,0)

        # Calculating the new position
        new_pos = (self.current_pos[0] + movement[0], self.current_pos[1] + movement[1])

        # Determining the type of square at the new position
        for square_type, positions in ego_state.items():
            if new_pos in positions:
                return square_type, new_pos

        # If new position does not match any known square type
        return "Unknown square type", new_pos

    def check_plan_one_step(self, action, state, env, agent_goals_and_actions):
        # use one step lookahead to see if action takes us to valid/intended location
        next_state_type, new_pos = self.check_next_state_type(state, action)
        goal_and_plan = agent_goals_and_actions[self.agent_id]
        subgoal = goal_and_plan['action_plan'][goal_and_plan['subgoal_num']]
        if next_state_type != 'ground' and new_pos != self.destination and action != 'FIRE_ZAP' and action[:8] != 'INTERACT' and subgoal[:7] != 'fire_at':
            # if next state is not ground, ie. collects unintended resource, replan with newly observed state information
            # update current subgoal with current position
            subgoal = goal_and_plan['action_plan'][goal_and_plan['subgoal_num']]
            # Splitting the subgoal into two parts at the first closing parenthesis
            part1, part2 = subgoal.split('),', 1)
            # Updating the first part with the agent's current position
            updated_part1 = part1[:part1.find('(') + 1] + str(self.current_pos)
            # Reassembling the updated subgoal
            subgoal = updated_part1 + ',' + part2
            goal_and_plan['action_plan'][goal_and_plan['subgoal_num']] = subgoal
            agent_goals_and_actions[self.agent_id] = goal_and_plan
            # make pathfinding grid include all resources excluding ones on the plan
            # Extracting coordinates from the action plans to exclude from grid
            waypoints = set()
            tuples = re.findall(r'\((\d+,\s*\d+)\)', subgoal)
            for tup in tuples:
                waypoints.add(tuple(map(int, tup.split(','))))
            waypoints = list(waypoints)
            # combine all known states
            self.combine_all_known_states(state) # update agent.all_known_states
            opponent_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
            labels = ['wall', 'green_box', 'red_box', opponent_key]
            plan_grid = env.build_grid_from_states(self.all_known_states, labels, waypoints)
            # empty actions queue and get new actions with new pos and new information
            while not self.all_actions.empty():
                self.all_actions.get()
            self.get_actions_from_plan(goal_and_plan, plan_grid, state)
            action = self.act()

        if action is None:
            print(f"Agent {self.agent_id} is None, choosing NOOP.")
            action = 'NOOP'

        return action, agent_goals_and_actions

    def combine_all_known_states(self, state):
        ego_state = state[self.agent_id]
        self.all_known_states = {}

        # wall locations are always known
        if 'global' in state:
            self.all_known_states['wall'] = set(state['global']['wall'])
        else:
            self.all_known_states['wall'] = set(state['wall'])

        # Add information from ego_state
        for key, coords in ego_state.items():
            if key != 'inventory' and key != 'wall':  # Exclude non-spatial data if needed
                self.all_known_states[key] = set(coords)

        # Add information from agent.memory_states
        for key, coords_with_step in self.memory_states.items():
            if key not in self.all_known_states:
                self.all_known_states[key] = set()
            for coord, _, _ in coords_with_step:
                self.all_known_states[key].add(coord)

        # Convert sets back to lists
        for key in self.all_known_states:
            self.all_known_states[key] = list(self.all_known_states[key])