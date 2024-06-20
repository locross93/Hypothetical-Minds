import abc
import ast
import numpy as np
from queue import Queue
from typing import List, Dict, Any, Optional

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
        player_key = self.agent_id
        opponent_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
        for entity_type in ['yellow_box', 'blue_box', 'purple_box', 'ground', player_key, opponent_key]:
            self.memory_states[entity_type] = []

    def generate_system_message(self):
        system_message = f"""
            You are Agent {self.agent_id} in the two player 'running_with_scissors_in_the_matrix__repeated'
            Melting Pot multiagent reinforcement learning environment that is an 23x15 (x by y) grid with resources to collect and walls to navigate around. 
            Players can move around the map and collect resources of 3 discrete types corresponding to rock, paper, and
            scissors strategies - Blue box = scissors - Purple box = paper - Yellow box = rock. 
            In addition to movement, the agents have an action to fire an "interaction" beam which initiates a duel 
            with one player getting positive reward and the other agent getting an opposite negative reward according to their inventories.
            All players carry an inventory with the count of resources picked up since last respawn and for each respawn start with an inventory of 1 resource each.
            This inventory is visible in the state with the key 'inventory'.
            To play a pure strategy, pick up about 5 resources of the color and then fire the interaction beam at the other player.
            To fire the interaction beam, use the fire_at(target_coord) action function, but you may typically want to move to the middle of the environmnent first to meet your opponent.
            State Description: This environment is partially-observable, you can observed a 5x5 grid around your agent depending on your position and orientation (you can see more in front of you than behind).
            Previously seen states will be represented in memory, but note that these states could potentially be outdated. For example the other agent could collect a resource that you previously saw.
            Given the partially-observable nature of the environment, you will need to explore the environment appropriately and select goals based on the information you've gathered.
            Also pay attention to your opponent's position when you see it in order to duel with them and gain information about their strategy.

            Objective:
            Your task is to devise efficient action plans for player {self.agent_id} to play a pure strategy against the other player.
            If you are player 0, you should play a pure scissors strategy (blue resources) and if you are player 1, you should play a pure paper strategy (purple resources).
            Consider the entire game state to plan the most efficient paths for resource collection and strategy execution.
            To do this you will need to think step by step about what actions to output in the following format for 
            these players to efficiently collect the appropriate resources/target inventories and play their strategy.
            Take into account the proximity of the target_coord from the src_coord and the shortest path to get to a target resource.

            ONLY USE THESE 2 ACTION FUNCTIONS:
            - move_to(src_coord, target_coord): Efficiently move agent from source coordinate to target coordinate. DO NOT move to any locations where walls are present.
            - fire_at(target_coord): Stay around specified coordinate and fire interaction when opponent is spotted to initiate duel. 
            After an interaction both players respawn.

            You should also specify which one of these potential goals your action plan corresponds to in a format that will be specified later.
            Mid-level goals:
            - Collect yellow resource
            - Collect blue resource
            - Collect purple resource
            - Seek out and duel with other player (use fire_at action func as the last action in the plan)
            - Explore location (use move_to action func, but DO NOT move to any locations where walls are present)

            Keep in mind that you will be prompted again with an updated game state after these action plans are executed.
            Therefore you do not need to plan everything at once and currently can output a subgoal or series of subgoals as you wish.
            """
        self.system_message = f"As the controller for agent {self.agent_id}, focus only on strategies for this specific agent. Ignore strategies for other players. {system_message}"
    
    def generate_initial_user_message(self, state, step):
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
        yellow_locations = ego_state.get('yellow_box', [])
        blue_locations = ego_state.get('blue_box', [])
        purple_locations = ego_state.get('purple_box', [])
        beam_locations = ego_state.get('beam', [])
        ground_locations = ego_state.get('ground', [])
        opponent_locations = [v[0] for k, v in ego_state.items() if k.startswith('player_1' if self.agent_id == 'player_0' else 'player_0')]

        strategy_request = f"""
            Strategy Request:
            Provide a strategy for player {self.agent_id} in Python dictionary format, parsable by `ast.literal_eval()`. 
            You are at step {step} of the game.
            Your response should be broken up into two parts:
                1. Reflection - based on the current state, what is your strategy or goal for your player, and how can this strategy be
                decomposed into a sequence of actions to efficiently achieve this goal? Think step by step about this. This could be fairly long.
                2. Action Plan - choose the first action for your player to begin this strategy?
                The output actions should be in the following format:
            Example of #2 Action Plan:
            ```python
            {{
              'goal': 'Explore location',
              'action_plans': ['move_to((17, 13), (7, 8))']
            }}
            ```
            The strategy should be efficient, considering the shortest paths to resources and strategic positioning for duels. 
            Format the dictionary as outlined below, listing the strategy and action plans.
            Do not use JSON or any other formatting. 
            Actions should align with the mid-level goals and action functions, \
            emphasizing efficient pathfinding and playing the corresponding strategies.
            Keep plan to one action function now at the first step. You will be prompted again when the action plan is finished and more information is observed.
            """     
        # - Wall Locations/Invalid move_to locations DO NOT MOVE TO ANY OF THESE LOCATIONS: {state['global']['wall']}   
        # - Observable Beam Locations: {beam_locations if 'beam' in ego_state else []}
        # - Observable Ground Locations: {ground_locations if 'ground' in ego_state else []}
        user_message = f"""Current State Description:
            - Global Map Size: {map_size} grid (Walls are located at the boundaries of the map and in other places that are invalid for move_to).
            - Valid Locations for move_to: {movable_locations}
            - Player Position: {player_position}
            - Player Orientation: {player_orientation}
            - Player Inventory (yellow, purple, blue): {player_inventory}
            - Egocentric Observations Size: 5x5 grid around your agent. You currently can observe the following based on your position and orientation:
            - Observable Yellow Box Locations: {yellow_locations if 'yellow_box' in ego_state else []}
            - Observable Blue Box Locations: {blue_locations if 'blue_box' in ego_state else []}
            - Observable Purple Box Locations: {purple_locations if 'purple_box' in ego_state else []}
            - Observable Opponent Locations: {opponent_locations}
            - Previously seen states from memory (format: ((x,y), step last observed): {self.memory_states}
            
            {strategy_request}
            """
        user_message = f"For agent {self.agent_id}: {user_message} Provide a strategy only for this agent, not for others."
        return user_message
    
    def generate_feedback_user_message(
            self, 
            state,
            execution_outcomes, 
            get_action_from_response_errors,
            rewards,
            step):
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
        #movable_locations = [v for k, v in state['global'].items() if k != 'wall']
        yellow_locations = ego_state.get('yellow_box', [])
        blue_locations = ego_state.get('blue_box', [])
        purple_locations = ego_state.get('purple_box', [])
        beam_locations = ego_state.get('beam', [])
        ground_locations = ego_state.get('ground', [])
        opponent_locations = [v[0] for k, v in ego_state.items() if k.startswith('player_1' if self.agent_id == 'player_0' else 'player_0')]
        
        # Preparing execution outcomes and rewards sections
        execution_outcomes_str = "\n".join(f"- {player}: {outcome}" for player, outcome in execution_outcomes.items())
        rewards_str = "\n".join(f"- {player}: {reward}" for player, reward in rewards.items())
        
        strategy_request = f"""
            Strategy Request:
            Provide a strategy for player {self.agent_id} in Python dictionary format, parsable by `ast.literal_eval()`. 
            You are at step {step} of the game.
            Your response should be broken up into two parts:
                1. Reflection - based on the current state, what is your strategy or goal for your player, and how can this strategy be
                decomposed into a sequence of actions to efficiently achieve this goal? Think step by step about this. This could be fairly long.
                2. Action Plan - what are the specific actions you would output for your player to execute this strategy?
                Output actions should be in the following format:
            Example of #2 Action Plan:
            ```python
            {{
              'goal': 'Collect blue resource',
              'action_plans': ['move_to((17, 13), (7, 8))', ...]
            }}
            ```
            The strategy should be efficient, considering the shortest paths to resources and strategic positioning for duels. 
            Format the dictionary as outlined below, listing the strategy and action plans.
            Do not use JSON or any other formatting. 
            Actions should align with the mid-level goals and action functions, \
            emphasizing efficient pathfinding and playing the corresponding strategies.
            Keep plans relatively short (<6 goals), especially at the early steps of an episode. You will be prompted again when the action plans are finished and more information is observed.
            """

        # - Wall Locations/Invalid move_to locations DO NOT MOVE TO ANY OF THESE LOCATIONS: {state['global']['wall']}   
        # - Observable Beam Locations: {beam_locations if 'beam' in ego_state else []}
        # - Observable Ground Locations: {ground_locations if 'ground' in ego_state else []}
        user_message = f"""Current State Description:
            - Global Map Size: {map_size} grid (Walls are located at the boundaries of the map and in other places that are invalid for move_to).
            - Valid Locations for move_to: {movable_locations}
            - Player Position: {player_position}
            - Player Orientation: {player_orientation}
            - Player Inventory (yellow, purple, blue): {player_inventory}
            - Egocentric Observations Size: 5x5 grid around your agent. You currently can observe the following based on your position and orientation:
            - Observable Yellow Box Locations: {yellow_locations if 'yellow_box' in ego_state else []}
            - Observable Blue Box Locations: {blue_locations if 'blue_box' in ego_state else []}
            - Observable Purple Box Locations: {purple_locations if 'purple_box' in ego_state else []}
            - Observable Opponent Locations: {opponent_locations}
            - Previously seen states from memory (format: ((x,y), step last observed): {self.memory_states}

            Execution Outcomes:
            {execution_outcomes_str}

            Error for extracting and executing actions from the response:
            {get_action_from_response_errors}

            Rewards:
            {rewards_str}

            {strategy_request}
            """
        return user_message

    def generate_interaction_feedback_user_message(
            self, 
            state,
            execution_outcomes, 
            get_action_from_response_errors,
            total_rewards,
            step_rewards,
            step):
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
        #movable_locations = [v for k, v in state['global'].items() if k != 'wall']
        yellow_locations = ego_state.get('yellow_box', [])
        blue_locations = ego_state.get('blue_box', [])
        purple_locations = ego_state.get('purple_box', [])
        beam_locations = ego_state.get('beam', [])
        ground_locations = ego_state.get('ground', [])
        opponent_locations = [v[0] for k, v in ego_state.items() if k.startswith('player_1' if self.agent_id == 'player_0' else 'player_0')]
        
        # Preparing execution outcomes and rewards sections
        execution_outcomes_str = "\n".join(f"- {player}: {outcome}" for player, outcome in execution_outcomes.items())
        rewards_str = "\n".join(f"- {player}: {reward}" for player, reward in total_rewards.items())
        
        strategy_request = f"""
            An interaction with the other player has occurred at step {step}. You received a reward of {step_rewards[self.agent_id]} for this interaction.
            If you were in the middle of an action plan, this plan is now terminated as the environment is reset after an interaction.
            Strategy Request:
            Provide the next strategy for player {self.agent_id} in Python dictionary format, parsable by `ast.literal_eval()`. 
            Your response should be broken up into two parts:
                1. Reflection - based on the current state and the result of this interaction, what is the next strategy or goal for your player, and how can this strategy be
                decomposed into a sequence of actions to efficiently achieve this goal? Think step by step about this. This could be fairly long.
                2. Action Plan - what are the specific actions you would output for your player to execute this strategy?
                Output actions should be in the following format:
            Example of #2 Action Plan:
            ```python
            {{
              'goal': 'Collect blue resource',
              'action_plans': ['move_to((17, 13), (7, 8))', ...]
            }}
            ```
            The strategy should be efficient, considering the shortest paths to resources and strategic positioning for duels. 
            Format the dictionary as outlined below, listing the strategy and action plans.
            Do not use JSON or any other formatting. 
            Actions should align with the mid-level goals and action functions, \
            emphasizing efficient pathfinding and playing the corresponding strategies.
            Keep plans relatively short (<6 goals), especially at the early steps of an episode. You will be prompted again when the action plans are finished and more information is observed.
            """

        # - Wall Locations/Invalid move_to locations DO NOT MOVE TO ANY OF THESE LOCATIONS: {state['global']['wall']}   
        # - Observable Beam Locations: {beam_locations if 'beam' in ego_state else []}
        # - Observable Ground Locations: {ground_locations if 'ground' in ego_state else []}
        user_message = f"""Current State Description:
            - Global Map Size: {map_size} grid (Walls are located at the boundaries of the map and in other places that are invalid for move_to).
            - Valid Locations for move_to: {movable_locations}
            - Player Position: {player_position}
            - Player Orientation: {player_orientation}
            - Player Inventory (yellow, purple, blue): {player_inventory}
            - Egocentric Observations Size: 5x5 grid around your agent. You currently can observe the following based on your position and orientation:
            - Observable Yellow Box Locations: {yellow_locations if 'yellow_box' in ego_state else []}
            - Observable Blue Box Locations: {blue_locations if 'blue_box' in ego_state else []}
            - Observable Purple Box Locations: {purple_locations if 'purple_box' in ego_state else []}
            - Observable Opponent Locations: {opponent_locations}
            - Previously seen states from memory (format: ((x,y), step last observed): {self.memory_states}

            Execution Outcomes:
            {execution_outcomes_str}

            Error for extracting and executing actions from the response:
            {get_action_from_response_errors}

            Total Rewards:
            {rewards_str}

            {strategy_request}
            """
        return user_message

    def extract_goals_and_actions(self, response):
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
            # Remove the common string escape sequences for new lines and tabs
            dict_str = dict_str.replace("\\n", "").replace("\\t", "")            
            # Convert the string representation of a dictionary into an actual dictionary
            goals_and_actions = ast.literal_eval(dict_str)
            return goals_and_actions
        except Exception as e:
            print(f"Error parsing goals and actions: {e}")
            return {}

    def get_actions_from_plan(
            self, 
            goals_and_actions: Dict[str, Any],
            grid: np.ndarray,
            state: Dict[str, Any]) -> Optional[str]:
        """Given a plan, return a list of actions to be performed by the agent."""
        try:
            self.goal = goals_and_actions['goal']
            self.action_plans = goals_and_actions['action_plans']
            self.subgoal_num = goals_and_actions['subgoal_num']
        except Exception as e:
            return f"Error parsing goals and actions: {e}" 
        #for action_plan in self.action_plans:
        # get actions for current subgoal
        action_plan = self.action_plans[self.subgoal_num]
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
            paths, actions, current_orient = func(start, goal, grid, self.orientation)
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
            self.all_actions.put(action)

    def is_valid_plan(self, state, goals_and_actions):
        """
        Check if the action plans contain any moves that lead to a wall.

        :param action_plans: List of strings representing the action plans.
        :param wall_locations: List of tuples representing the wall locations.
        :return: Boolean indicating whether the plans are valid or not.
        """
        wall_locations = state['global']['wall']
        for plan in goals_and_actions['action_plans']:
            # Extract the destination location from the action plan
            destination = tuple(map(int, plan.split('(')[-1].strip(')').split(', ')))
            
            # Check if the destination is a wall location
            if destination in wall_locations:
                response = f"Invalid plan as it leads to a wall: {destination}. Replan and try again. DO NOT INCLUDE MOVE_TO ACTIONS THAT LEAD TO WALLS. \
                Think step by step about whether target locations are valid. "
                return False, response  # Invalid plan as it leads to a wall

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
                for action_plan in self.action_plans:
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
                for action_plan in self.action_plans:
                    if 'move_to' in action_plan:
                        split_idx = action_plan.find('(')
                        func_args = ast.literal_eval(action_plan[split_idx+1:-1])
                        goal_pos = func_args[1]                                
                        output = f"Out of game now due to hit by laser in the last 5 steps."
                        return output

    def update_memory(self, state, step):
        """
        Update the memory states with the latest observations.

        :param state: Dictionary of the current state.
        :param step: The current step number in the game.
        """
        ego_state = state[self.agent_id]
        player_key = self.agent_id
        opponent_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
        for entity_type in ['yellow_box', 'blue_box', 'purple_box', 'ground', player_key, opponent_key]:
            if entity_type[:6] == 'player':
                observed_locations = [v[0] for k, v in ego_state.items() if k.startswith(entity_type)]
            else:
                observed_locations = ego_state.get(entity_type, [])
            for location in observed_locations:
                # Remove this location from all other entity types
                for other_entity in self.memory_states:
                    if other_entity != entity_type:
                        self.memory_states[other_entity] = [
                            (loc, step_str) for loc, step_str in self.memory_states[other_entity] if loc != location
                        ]

                # Update or add the location with the latest step number and remove older references of the same location
                self.memory_states[entity_type] = [
                    (loc, step) for loc, step in self.memory_states[entity_type] if loc != location
                ]
                self.memory_states[entity_type].append((location, 'Step: '+str(step)))

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
        #breakpoint()

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

    def combine_all_known_states(self, state):
        ego_state = state[self.agent_id]
        self.all_known_states = {}

        # wall locations are always known
        self.all_known_states['wall'] = set(state['global']['wall'])

        # Add information from ego_state
        for key, coords in ego_state.items():
            if key != 'inventory' and key != 'wall':  # Exclude non-spatial data if needed
                self.all_known_states[key] = set(coords)

        # Add information from agent.memory_states
        for key, coords_with_step in self.memory_states.items():
            if key not in self.all_known_states:
                self.all_known_states[key] = set()
            for coord, _ in coords_with_step:
                self.all_known_states[key].add(coord)

        # Convert sets back to lists
        for key in self.all_known_states:
            self.all_known_states[key] = list(self.all_known_states[key])



        