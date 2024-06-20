import abc
import ast
import asyncio
from copy import deepcopy
import numpy as np
from queue import Queue
from typing import List, Dict, Any, Tuple, Optional

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
        self.pot_state = {}
        for location in [(4, 2), (4, 3)]:
            self.pot_state[location] = {"num_tomatoes": 0, "progress": 0, "done": False, "player_action": False}
        player_key = self.agent_id
        teammate_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
        #for entity_type in ['tomato_dispenser', 'pot_0_tomatoes', 'delivery_location', 'dish_dispenser', 'ground', player_key, teammate_key]:
        for entity_type in ['accessible_tomato_dispenser', 'inaccessible_tomato_dispenser', 'accessible_delivery_location', 'inaccessible_delivery_location', 'accessible_dish_dispenser', 'inaccessible_dish_dispenser', 'accessible_counter', 'pot', teammate_key]:
            self.memory_states[entity_type] = []

    def generate_system_message(self):
        self.system_message = f"""
        You are Player {self.agent_id} in the Collaborative Cooking Asymmetric environment, the goal is to cook and deliver tomato soup dishes with a partner.
        The environment consists of a kitchen with a tomato dispenser, pots, delivery locations, and dish dispensers.
        Each agent (of 2) has access to specific parts of the kitchen and can perform actions like picking up ingredients, putting soup in a dish, and delivering cooked soup dishes.
        There is an impassable barrier in the middle of the kitchen that separates the agents' sides at x=4, where the pots are located.
        The goal is to work together with the other agent to efficiently cook and serve as many dishes of tomato soup as possible to maximize the collective reward.
        However, communication is not possible, so you must infer your partner's strategy from their actions and adapt accordingly to coordinate tasks.
        To cook tomato soup, 1. put 3 tomatoes in a pot, 2. pick up a dish when it is finished cooking, 3. put the cooked soup in a dish, and 4. deliver it to the delivery location.
        Your team receives a reward of 20 for each successfully delivered dish.
        Only interact with objects on your side of the kitchen.
        You can only hold one tomato at once.
        You cannot pick up a tomato from the tomato dispenser with another item like a dish in your hand.
        The environment is partially observable, and you can only see a 5x5 grid around your agent.
        You will be prompted at different points to provide high-level strategies and lower-level action plans to achieve them.
        Use these three functions for lower-level action plans:
        - move_to(src_coord, target_coord): Efficiently move agent from source coordinate to target coordinate. Only move to valid move_to locations where counters or objects are not present. Use sparingly.
        - interact(target_coord): Move to and interact with the entity at the target coordinate, such as picking up ingredients or delivering dishes of cooked soup. To place an object down on a counter to free your hands, use interact(counter_coord). Mostly use this function. 
        - wait(target_coord): Wait for the pot at target_coord to finish cooking. Check the progress of the pots and only use valid locations where pots are present. You probably only want to use this when both pots are full to maximize efficiency.
        Most of the time you will just want to use the interact function because it both moves to and interacts with objects, therefore all the cooking steps can be completed with the interact function.
        To put down an item to pick something else up, interact with a counter to free your hands. Do not put down items on the floor or the delivery location.
        """

    def get_state_info(self, state, step):
        ego_state = state[self.agent_id]
        player_position = {k: v for k, v in ego_state.items() if k.startswith(self.agent_id)}
        currently_holding = next(('tomato' if 'tomato' in key else 'dish' if 'dish' in key else 'cooked_soup' if 'cooked_soup' in key else 'nothing' for key in player_position), 'nothing')
        state_info_dict = {
            'step': step,
            self.agent_id: {'position': player_position, 'holding': currently_holding},
        }
        for location, pot in self.pot_state.items():
            state_info_dict['Pot '+str(location)] = {'num_tomatoes': pot['num_tomatoes'], 'progress': str(pot['progress'])+'/10', 'done': pot['done']}

        return state_info_dict

    def generate_feedback_user_message(
            self,
            state,
            execution_outcomes,
            get_action_from_response_errors,
            evaluator_response,
            rewards,
            step):
        ego_state = state[self.agent_id]
        # Extracting information from the state
        map_size = "9x5"
        player_position = {k: v for k, v in ego_state.items() if k.startswith(self.agent_id)}
        player_x = [v for k, v in ego_state.items() if k.startswith('player_0')][0][0][0]
        self.player_spot = 'left' if player_x < 4 else 'right'
        player_orientation = list(player_position.keys())[0].split('-')[-1]
        currently_holding = next(('tomato' if 'tomato' in key else 'empty_dish' if 'dish' in key else 'cooked_soup' if 'cooked_soup' in key else 'nothing' for key in player_position), 'nothing')
        
        accessible_tomato_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('tomato_dispenser') and (v[0][0] < 4 if self.player_spot == 'left' else v[0][0] > 4)]
        inaccessible_tomato_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('tomato_dispenser') and (v[0][0] > 4 if self.player_spot == 'left' else v[0][0] < 4)]
        accessible_delivery_locations = [v[0] for k, v in ego_state.items() if k.startswith('delivery') and (v[0][0] < 4 if self.player_spot == 'left' else v[0][0] > 4)]
        inaccessible_delivery_locations = [v[0] for k, v in ego_state.items() if k.startswith('delivery') and (v[0][0] > 4 if self.player_spot == 'left' else v[0][0] < 4)]
        accessible_dish_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('dish_dispenser') and (v[0][0] < 4 if self.player_spot == 'left' else v[0][0] > 4)]
        inaccessible_dish_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('dish_dispenser') and (v[0][0] > 4 if self.player_spot == 'left' else v[0][0] < 4)]
        left_acc_counters = [(2, 4), (1, 4), (2, 1), (1, 0), (0,3), (0,2)]
        right_acc_counters = [(6, 1), (6, 4), (7, 4), (7, 0), (8, 2), (8, 3)]
        accessible_counter_locations = left_acc_counters if self.player_spot == 'left' else right_acc_counters
        accessible_counter_locations = [v for v in accessible_counter_locations if v in ego_state['counter']]
        teammate_locations = [v[0] for k, v in ego_state.items() if k.startswith('player_1' if self.agent_id == 'player_0' else 'player_0')]

        # Format pot state information for the user message
        pot_state_str = "\n".join(
            f"- Pot at {location}: Number of Tomatoes: {pot['num_tomatoes']}, Progress: {pot['progress']}/10, Done: {pot['done']}, Player action: {pot['player_action']}"
            for location, pot in self.pot_state.items()
        )

        rewards_str = "\n".join(f"- {player}: {reward}" for player, reward in rewards.items())

        strategy_request = f"""
        Strategy Request:
        You are at step {step} of the game.
        Your task is to devise efficient action plans for agent {self.agent_id}, reason through what the next subgoals should be given the state information.
        Your response should be broken up into two parts:
            1. Subgoal Plan - based on the current state plan a sequence of subgoals and actions. 
            For every subgoal, think step by step about the best action function and parameter to use for that function. This could be fairly long.
            2. Action Plan - output this sequence of actions in the following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python. 
            Make sure the action plan uses the action functions described below appropriately. In particular, before interacting with an object, you will first need to move to the closest ground location next to the target coordinate, not the target coordinate itself.
        Example response:
        Subgoal Plan: Given the current state and my high-level strategy to focus on cooking tomato soup dishes, I should:
        Move to the tomato dispenser and pick up a tomato.
        ```python
        {{
          'action_plan': ['interact((5, 1))']
        }}
        ```
        The strategy should be efficient, considering the shortest paths and effective coordination with the other agent.
        Format the dictionary as outlined above, listing the action plan.
        Do not use JSON or any other formatting.
        Actions should align with the action functions, emphasizing efficient pathfinding and collaboration with other agents.
        Consider the entire game state to plan the most efficient paths and coordinate actions with other agents.
        
        ONLY USE THESE 3 ACTION FUNCTIONS:
            - move_to(src_coord, target_coord): Efficiently move agent from source coordinate to target coordinate. Only move to valid move_to locations where counters or objects are not present.
            - interact(target_coord): Move to and interact with the entity at the target coordinate, such as picking up ingredients or delivering dishes of cooked soup. To place an object on a counter to free your hands, use interact(counter_coord).
            - wait(target_coord): Wait for the pot at target_coord to finish cooking. Check the progress of the pots and only use valid locations where pots are present. You probably only want to use this when both pots are full to maximize efficiency.
            Most of the time you will just use this interact function.
            To interact with an object, output the coordinate where that object is located.
            For example to pick up a tomato from a tomato dispenser at (5, 1), you would output 'action_plan': ['interact((5, 1))']
            Or to pick up a tomato from a tomato dispenser at (0, 1), you would output 'action_plan': ['interact((0, 1))']
            To interact with a pot at (4, 2), you would output 'action_plan': ['interact((4, 2))']
            To put a dish down on a counter at (2, 4) to free your hands, you would output 'action_plan': ['interact((2, 4))']
            Moreover, only select valid location coordinates for objects on your side of the room. 
            Pay attention to how the counters and pots block your path to the other side of the room.
            Use move_to only when you need to explore the environment to find objects or to see what your teammate is doing.  
            If you need to put down an item to pick something else up, interact with a counter to free your hands.

        Keep plans relatively short (<4 subgoals), especially at the early steps of an episode. You will be prompted again when the action plans are finished and more information is observed.
        """

        user_message = f"""
        Current State Description:
        - Global Map Size: {map_size} grid (Counters are located at the boundaries of the map and in other places that are invalid for move_to).
        - Player Position: {player_position}
        - Player Orientation: {player_orientation}
        - Player Currently Holding: {currently_holding}
        - Egocentric Observations Size: 5x5 grid around your agent. You currently can observe the following based on your position and orientation:
        {pot_state_str}
        - Observable Accessible Tomato Dispenser Locations: {accessible_tomato_dispenser_locations}
        - Observable Inaccessible Tomato Dispenser Locations: {inaccessible_tomato_dispenser_locations}
        - Observable Accessible Delivery Locations: {accessible_delivery_locations}
        - Observable Inaccessible Delivery Locations: {inaccessible_delivery_locations}
        - Observable Accessible Dish Dispenser Locations: {accessible_dish_dispenser_locations}
        - Observable Inaccessible Dish Dispenser Locations: {inaccessible_dish_dispenser_locations}
        - Observable Accessible Counter Locations: {accessible_counter_locations}
        - Observable Location of the Other Agent: {teammate_locations}
        - Previously seen states from memory (format: ((x,y), step last observed): {self.memory_states}

        Execution Outcomes:
        {execution_outcomes}

        Error for extracting and executing actions from the response:
        {get_action_from_response_errors}

        Rewards:
        {rewards_str}

        {strategy_request}

        Feedback from Evaluator:
        {evaluator_response}
        """

        return user_message

    async def evaluate_action_outcomes(self, state, goal_and_plan, state_info_dict_list, reward_during_plan, subgoal_failed=False):
        if subgoal_failed:
            user_message_preamble = f"""
            You are an action plan evaluator. 
            The last subgoal included an interact action that failed.
            Your task is to look at the subgoal the agent took, the state of the environment before and after the subgoal, 
            and evaluate why the subgoal was unsuccessful and provide feedback about what the agent should do next time.
            We will next plan an entire new action plan, so suggest specific action plans and action functions to use next when applicable.
            """
        else:
            user_message_preamble = f"""
            You are an action plan evaluator. 
            Your task is to look at the action plan the agent took, the state of the environment before the plan and the state of the environment after the plan, 
            and evaluate whether the action plan was successful, and if not, provide feedback about what failed and what the agent should do next time.
            Take into account that your teammate could have influenced the outcome of the subgoal in some circumstances.
            Suggest specific action plans and action functions to use next when applicable.
            """

        user_message_outcomes = f"""
        State Before Plan: {state_info_dict_list[-2]}
        Action Plan: {goal_and_plan}
        State After Plan: {state_info_dict_list[-1]}
        Reward received during plan: {reward_during_plan}
        """

        ego_state = state[self.agent_id]
        player_x = [v for k, v in ego_state.items() if k.startswith('player_0')][0][0][0]
        self.player_spot = 'left' if player_x < 4 else 'right'
        accessible_tomato_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('tomato_dispenser') and (v[0][0] < 4 if self.player_spot == 'left' else v[0][0] > 4)]
        inaccessible_tomato_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('tomato_dispenser') and (v[0][0] > 4 if self.player_spot == 'left' else v[0][0] < 4)]
        accessible_delivery_locations = [v[0] for k, v in ego_state.items() if k.startswith('delivery') and (v[0][0] < 4 if self.player_spot == 'left' else v[0][0] > 4)]
        inaccessible_delivery_locations = [v[0] for k, v in ego_state.items() if k.startswith('delivery') and (v[0][0] > 4 if self.player_spot == 'left' else v[0][0] < 4)]
        accessible_dish_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('dish_dispenser') and (v[0][0] < 4 if self.player_spot == 'left' else v[0][0] > 4)]
        inaccessible_dish_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('dish_dispenser') and (v[0][0] > 4 if self.player_spot == 'left' else v[0][0] < 4)]
        left_acc_counters = [(2, 4), (1, 4), (2, 1), (1, 0), (0,3), (0,2)]
        right_acc_counters = [(6, 1), (6, 4), (7, 4), (7, 0), (8, 2), (8, 3)]
        accessible_counter_locations = left_acc_counters if self.player_spot == 'left' else right_acc_counters
        accessible_counter_locations = [v for v in accessible_counter_locations if v in ego_state['counter']]
        teammate_locations = [v[0] for k, v in ego_state.items() if k.startswith('player_1' if self.agent_id == 'player_0' else 'player_0')]

        user_message_ego = f"""
        Egocentric Observations Size: 5x5 grid around your agent. You currently can observe the following based on your position and orientation:
        - Observable Accessible Tomato Dispenser Locations: {accessible_tomato_dispenser_locations}
        - Observable Inaccessible Tomato Dispenser Locations: {inaccessible_tomato_dispenser_locations}
        - Observable Accessible Delivery Locations: {accessible_delivery_locations}
        - Observable Inaccessible Delivery Locations: {inaccessible_delivery_locations}
        - Observable Accessible Dish Dispenser Locations: {accessible_dish_dispenser_locations}
        - Observable Inaccessible Dish Dispenser Locations: {inaccessible_dish_dispenser_locations}
        - Observable Accessible Counter Locations: {accessible_counter_locations}
        - Observable Location of the Other Agent: {teammate_locations}
        - Previously seen states from memory (format: ((x,y), step last observed): {self.memory_states}
        """

        user_message = user_message_preamble + user_message_outcomes + user_message_ego

        response = await asyncio.gather(
            *[self.controller.async_batch_prompt(self.system_message, [user_message])]
        )
        return user_message_outcomes, response[0][0]

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
            state: Dict[str, Any]) -> Optional[str]:
        """Given a plan, return a list of actions to be performed by the agent."""
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
        if func_name != 'wait':            
            func = getattr(action_funcs, func_name)
        else:
            actions = [action_plan]   
      
        if func_name == 'move_to':                
            start, goal = func_args                  
            paths, actions, current_orient, path_found = func(start, goal, grid, self.orientation)

            # update agent's position (if moved) and orientation
            if len(paths) > 0:
                self.pos = paths[-1]
            self.orientation = deepcopy(current_orient)
            self.destination = deepcopy(goal)
        elif func_name == 'interact':
            target = func_args
            self.destination = deepcopy(target)
            actions, current_orient = func(self.pos, self.orientation, target, grid)
            self.orientation = current_orient

        assert(self.orientation in ['N', 'E', 'S', 'W'])

        for action in actions:                      
            self.all_actions.put(action)

    def build_grid_from_states(self, states: Dict[str, List[Tuple[int, int]]], labels: List[str], ignore: List[Tuple[int, int]] = None) -> np.ndarray:
        """Build a grid from a dictionary of states. Setting walls and other obstacles you pass in (as str in labels) to 1, 
        and all other things as 0."""
        grid_width = 9
        grid_height = 5
        grid = np.zeros((grid_width, grid_height))
        for label, coords in states.items():
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
            
        #counter_locations = state['global']['counter']
        movable_locations = []
        for k, v in state['global'].items():
            # accessible ground locations on their part of the kitchen
            if k.startswith('ground'):
                for loc in v:
                    if self.player_spot == 'left' and loc[0] < 4:
                        movable_locations.append(loc)
                    elif self.player_spot == 'right' and loc[0] > 4:
                        movable_locations.append(loc)

            if k.startswith(self.agent_id):
                for loc in v:
                    movable_locations.append(loc)

        interactable_locations = []
        for k, v in state['global'].items():
            if k.startswith('tomato_dispenser') or k.startswith('dish_dispenser') or k.startswith('delivery') or k.startswith('pot') or k.startswith('counter'):
                for loc in v:
                    if self.player_spot == 'left' and loc[0] <= 4:
                        interactable_locations.append(loc)
                    elif self.player_spot == 'right' and loc[0] >= 4:
                        interactable_locations.append(loc)

        waitable_locations = []
        for k, v in state['global'].items():
            if k.startswith('pot'):
                for loc in v:
                    waitable_locations.append(loc)

        response = None
        valid_plan = True
        for plan in goals_and_actions['action_plan']:
            # Extract the destination location from the action plan
            try:
                destination = tuple(map(int, plan.split('(')[-1].strip(')').split(', ')))
            except ValueError:
                response = f"Invalid destination location in action plan: {plan}. Replan and try again."
                return False, response
            split_idx = plan.find('(')
            func_name = plan[:split_idx]
            
            # Check if the destination is a counter location
            if func_name == 'move_to' and destination not in movable_locations:
                response = f"Invalid plan as it leads to a immovable location: {destination}. Replan and try again. DO NOT INCLUDE MOVE_TO ACTIONS THAT LEAD TO COUNTERS, OBJECTS, OR TO THE OTHER SIDE OF THE KITCHEN. \
                Think step by step about whether target locations are valid. Only select valid Locations coordinates for the move_to function that are listed."
                valid_plan = False

            # Check if the destination is an interactable location
            if func_name == 'interact' and destination not in interactable_locations:
                response = f"Invalid plan as it leads to a non-interactable location: {destination}. Replan and try again. DO NOT INCLUDE INTERACT ACTIONS THAT LEAD TO LOCATIONS WITHOUT OBJECTS OR TO THE OTHER SIDE OF THE KITCHEN. \
                Think step by step about whether target locations are valid. Only select valid Locations coordinates for the interact function that are listed."
                valid_plan = False

            # Check if the destination is a waitable location
            if func_name == 'wait' and destination not in waitable_locations:
                response = f"Invalid plan as it leads to a non-waitable location: {destination}. Replan and try again. DO NOT INCLUDE WAIT ACTIONS THAT LEAD TO LOCATIONS WITHOUT POTS OR TO THE OTHER SIDE OF THE KITCHEN. \
                Think step by step about whether target locations are valid. Only select valid Locations coordinates for the wait function that are listed."
                valid_plan = False

        if not valid_plan:
            feedback_message = f"""
                Invalid plan that was previously generated, create a new valid plan and try again.:
                {goals_and_actions}
                """
            feedback_message = feedback_message + response
        else:
            feedback_message = f"Valid plan: {goals_and_actions}"

        return valid_plan, feedback_message

    def act(self) -> Optional[str]:
        """Return the next action to be performed by the agent. 
        If no action is available, return None.
        """
        if not self.all_actions.empty():        
            return self.all_actions.get()

    def update_state(self, state: Dict[str, Any], step: int) -> Optional[str]:
        # Extract pot state information
        ego_state = state[self.agent_id]
        for key, locations in ego_state.items():
            if key.startswith("pot_"):
                for location in locations:
                    if "tomatoes" in key:
                        pot_info = key.split("_")
                        self.pot_state[location]["num_tomatoes"] = int(pot_info[1])
                        if "done" in key:
                            self.pot_state[location]["progress"] = 10
                            self.pot_state[location]["done"] = True
                        elif "progress" in key:
                            progress_index = pot_info.index("progress") + 1
                            self.pot_state[location]["progress"] = int(pot_info[progress_index])
                        else:
                            self.pot_state[location]["progress"] = 0
                            self.pot_state[location]["done"] = False
                    else:
                        self.pot_state[location]["num_tomatoes"] = 0
                        self.pot_state[location]["progress"] = 0
                        self.pot_state[location]["done"] = False
                    
                    if "_p" in key:
                        self.pot_state[location]["player_action"] = True

        """Update the position of the agent."""
        try:
            agent_key = [item for item in state['global'].keys() if item.startswith(self.agent_id)][0]        
            self.pos = state['global'][agent_key][0]    # (x, y) -> (col, row)                    
            self.orientation = agent_key.split('-')[1]            
            assert(self.orientation in ['N', 'E', 'S', 'W'])
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

    def update_memory(self, state, step, action):
        """
        Update the memory states with the latest observations.

        :param state: Dictionary of the current state.
        :param step: The current step number in the game.
        """
        ego_state = state[self.agent_id]
        player_key = self.agent_id
        teammate_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
        player_x = [v for k, v in ego_state.items() if k.startswith('player_0')][0][0][0]
        self.player_spot = 'left' if player_x < 4 else 'right'

        accessible_tomato_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('tomato_dispenser') and (v[0][0] < 4 if self.player_spot == 'left' else v[0][0] > 4)]
        inaccessible_tomato_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('tomato_dispenser') and (v[0][0] > 4 if self.player_spot == 'left' else v[0][0] < 4)]
        accessible_delivery_locations = [v[0] for k, v in ego_state.items() if k.startswith('delivery') and (v[0][0] < 4 if self.player_spot == 'left' else v[0][0] > 4)]
        inaccessible_delivery_locations = [v[0] for k, v in ego_state.items() if k.startswith('delivery') and (v[0][0] > 4 if self.player_spot == 'left' else v[0][0] < 4)]
        accessible_dish_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('dish_dispenser') and (v[0][0] < 4 if self.player_spot == 'left' else v[0][0] > 4)]
        inaccessible_dish_dispenser_locations = [v[0] for k, v in ego_state.items() if k.startswith('dish_dispenser') and (v[0][0] > 4 if self.player_spot == 'left' else v[0][0] < 4)]
        left_acc_counters = [(2, 4), (1, 4), (2, 1), (1, 0), (0,3), (0,2)]
        right_acc_counters = [(6, 1), (6, 4), (7, 4), (7, 0), (8, 2), (8, 3)]
        accessible_counter_locations = left_acc_counters if self.player_spot == 'left' else right_acc_counters
        accessible_counter_locations = [v for v in accessible_counter_locations if v in ego_state['counter']]
        pot_locations = [v[0] for k, v in ego_state.items() if k.startswith('pot')]
        teammate_locations = [v[0] for k, v in ego_state.items() if k.startswith('player_1' if self.agent_id == 'player_0' else 'player_0')]

        for entity_type, observed_locations in [
            ('accessible_tomato_dispenser', accessible_tomato_dispenser_locations),
            ('inaccessible_tomato_dispenser', inaccessible_tomato_dispenser_locations),
            ('accessible_delivery_location', accessible_delivery_locations),
            ('inaccessible_delivery_location', inaccessible_delivery_locations),
            ('accessible_dish_dispenser', accessible_dish_dispenser_locations),
            ('inaccessible_dish_dispenser', inaccessible_dish_dispenser_locations),
            ('accessible_counter', accessible_counter_locations),
            ('pot', pot_locations),
            (player_key, ego_state.get(player_key, [])),
            (teammate_key, teammate_locations)
        ]:
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

    def check_next_state_type(self, state, action):
        # used to determine move outcome based on orientation
        action_outcome_dict = {
            'N': {'FORWARD': (0, -1), 'STEP_LEFT': (-1, 0), 'STEP_RIGHT': (1, 0), 'BACKWARD': (0, 1),
                'TURN_LEFT': 'W', 'TURN_RIGHT': 'E', 'INTERACT': 'N', 'NOOP': (0, 0)},
            'E': {'FORWARD': (1, 0), 'STEP_LEFT': (0, -1), 'STEP_RIGHT': (0, 1), 'BACKWARD': (-1, 0),
                'TURN_LEFT': 'N', 'TURN_RIGHT': 'S', 'INTERACT': 'E', 'NOOP': (0, 0)},
            'S': {'FORWARD': (0, 1), 'STEP_LEFT': (1, 0), 'STEP_RIGHT': (-1, 0), 'BACKWARD': (0, -1),
                'TURN_LEFT': 'E', 'TURN_RIGHT': 'W', 'INTERACT': 'S', 'NOOP': (0, 0)},
            'W': {'FORWARD': (-1, 0), 'STEP_LEFT': (0, 1), 'STEP_RIGHT': (0, -1), 'BACKWARD': (1, 0),
                'TURN_LEFT': 'S', 'TURN_RIGHT': 'N', 'INTERACT': 'W', 'NOOP': (0, 0)},
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

        # counter locations are always known
        self.all_known_states['counter'] = set(state['global']['counter'])

        # Add information from ego_state
        for key, coords in ego_state.items():
            if key != 'counter':  # Exclude non-spatial data if needed
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