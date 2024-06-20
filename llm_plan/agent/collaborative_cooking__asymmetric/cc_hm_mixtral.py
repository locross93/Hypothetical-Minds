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
        self.pot_state_memory = []
        self.teammate_strategy = ''
        self.teammate_hypotheses = {}
        self.teammate_actions = []
        self.prev_ego_states = []
        self.delivery_num = 0
        self.hypothesis_num = 0
        self.good_hypothesis_found = False
        self.alpha = 0.3  # learning rate for updating hypothesis values
        self.correct_guess_reward = 1.0
        self.good_hypothesis_thr = 0.7
        self.top_k = 3  # number of top hypotheses to evaluate
        self.self_improve = config['self_improve']
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
        You need to pick up a dish before you pick up cooked soup from a pot.
        The environment is partially observable, and you can only see a 5x5 grid around your agent.
        You will be prompted at different points to provide high-level strategies and lower-level action plans to achieve them.
        Use these three functions for lower-level action plans:
        - move_to(src_coord, target_coord): Efficiently move agent from source coordinate to target coordinate. Only move to valid move_to locations where counters or objects are not present. Use sparingly.
        - interact(target_coord): Move to and interact with the entity at the target coordinate, such as picking up ingredients or delivering dishes of cooked soup. To place an object down on a counter to free your hands, use interact(counter_coord). Mostly use this function. 
        - wait(target_coord): Wait for the pot at target_coord to finish cooking. Check the progress of the pots and only use valid locations where pots are present. You probably only want to use this when both pots are full to maximize efficiency.
        Most of the time you will just want to use the interact function because it both moves to and interacts with objects, therefore all the cooking steps can be completed with the interact function.
        To put down an item to pick something else up, interact with a counter to free your hands. Do not put down items on the floor or the delivery location.
        """

    def generate_hls_user_message(self, state, step):
        ego_state = state[self.agent_id]
        # Extracting information from the state
        map_size = "9x5"
        player_position = {k: v for k, v in ego_state.items() if k.startswith(self.agent_id)}
        player_x = [v for k, v in ego_state.items() if k.startswith('player_0')][0][0][0]
        self.player_spot = 'left' if player_x < 4 else 'right'
        player_orientation = list(player_position.keys())[0].split('-')[-1]
        currently_holding = next(('tomato' if 'tomato' in key else 'empty_dish' if 'dish' in key else 'cooked_soup' if 'cooked_soup' in key else 'nothing' for key in player_position), 'nothing')

        pot_locations = [v[0] for k, v in ego_state.items() if k.startswith('pot')]
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

        if step == 0:
            prefix = f"""
                Strategy Request:
                You are at step {step} of the game.
                Provide a strategy for agent {self.agent_id}.
                Your response should outline a high-level strategy - what strategy do you want to take first and why?
                """
        else:
            prefix = f"""
                Strategy Request:
                You are at step {step} of the game.
                Provide a strategy for agent {self.agent_id}.
                Your response should outline a high-level strategy - what strategy do you want to take next and why?
                Factor in your current hypothesis about what your teammate is doing and what their strategy is: {self.teammate_strategy}
                Think step by step about how to adapt to their behavior and maximize all resources and efficiency accordingly
                """

        strategy_request = f"""Your strategy should outline an algorithm or list of principles to follow given certain conditions in the environment in order to maximize collective reward.
        This response will be shown to you in the future in order for you to select lower-level actions to implement this strategy.
        Example response:
        High-level strategy: I want to focus on cooking tomato soup dishes.
        You will be prompted again shortly to select subgoals and action plans to execute this strategy, so do not include that in your response yet.
        """
        strategy_request = prefix + strategy_request

        user_message = f"""
        Current State Description:
        - Global Map Size: {map_size} grid (Counters are located at the boundaries of the map and in other places that are invalid for move_to).
        - Player Position: {player_position}
        - Player Orientation: {player_orientation}
        - Player Currently Holding: {currently_holding}
        - Egocentric Observations Size: 5x5 grid around your agent. You currently can observe the following based on your position and orientation:
        - Observable Pot Locations: {pot_locations}
        - Observable Accessible Tomato Dispenser Locations: {accessible_tomato_dispenser_locations}
        - Observable Inaccessible Tomato Dispenser Locations: {inaccessible_tomato_dispenser_locations}
        - Observable Accessible Delivery Locations: {accessible_delivery_locations}
        - Observable Inaccessible Delivery Locations: {inaccessible_delivery_locations}
        - Observable Accessible Dish Dispenser Locations: {accessible_dish_dispenser_locations}
        - Observable Inaccessible Dish Dispenser Locations: {inaccessible_dish_dispenser_locations}
        - Observable Accessible Counter Locations: {accessible_counter_locations}
        - Observable Location of the Other Agent: {teammate_locations}
        - Previously seen states from memory (format: ((x,y), step last observed): {self.memory_states}

        {strategy_request}
        """

        return user_message

    def get_state_info(self, state, step):
        ego_state = state[self.agent_id]
        player_position = {k: v for k, v in ego_state.items() if k.startswith(self.agent_id)}
        currently_holding = next(('tomato' if 'tomato' in key else 'empty_dish' if 'dish' in key else 'cooked_soup' if 'cooked_soup' in key else 'nothing' for key in player_position), 'nothing')
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
        Your previously specified high-level strategy is: {self.my_strategy}
        Your response should be broken up into two parts:
            1. Subgoal Plan - based on the current state and the high-level strategy you previously specified,
            decompose this strategy into a sequence of subgoals and actions to efficiently implement this strategy. 
            For every subgoal, think step by step about the best action function and parameter to use for that function. This could be fairly long.
            2. Action Plan - output this sequence of actions in the following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python. 
            Make sure the action plan uses the action functions described below appropriately. In particular, before interacting with an object, you will first need to move to the closest ground location next to the target coordinate, not the target coordinate itself.
        Example response 1:
        Subgoal Plan: Given the current state and my high-level strategy to focus on cooking tomato soup dishes, I should:
        Move to the tomato dispenser and pick up a tomato.
        ```python
        {{
          'action_plan': ['interact((5, 1))']
        }}
        ```
        Example response 2:
        Subgoal Plan: Given the current state and my high-level strategy to focus on delivering tomato soup dishes, I should:
        Move to the dish dispenser and pick up a dish, then plate the cooked soup.
        ```python
        {{
          'action_plan': ['interact((3, 4))', 'interact((4, 2))']
        }}
        ```
        Example response 3:
        Subgoal Plan: Next I should move to the delivery location and deliver the cooked soup.
        ```python
        {{
          'action_plan': ['interact((3, 1))']
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
        - Teammate Hypothesis: {self.teammate_strategy}
        """

        user_message = user_message_preamble + user_message_outcomes + user_message_ego

        response = await asyncio.gather(
            *[self.controller.async_batch_prompt(self.system_message, [user_message])]
        )
        response = response[0]
        return user_message_outcomes, response[0][0]

    def infer_teammate_strategy(self, step):
        # get the top N hypotheses
        sorted_keys = sorted([key for key in self.teammate_hypotheses],
                    key=lambda x: self.teammate_hypotheses[x]['value'], 
                    reverse=True)
        top_keys = sorted_keys[:self.top_k]
        # show top hypotheses with value > 0
        self.top_hypotheses = {key: self.teammate_hypotheses[key] for key in top_keys if self.teammate_hypotheses[key]['value'] > 0}
        
        # Prepare the user message
        if self.self_improve:
            user_message = f"""
            Based on the observed actions of your teammate (player_1), what do you think their strategy is?
            Are they specializing in any specific activity or subtask?

            Teammate's observed actions:
            {self.teammate_actions}

            Here are your previous hypotheses about the strategy your partner is playing: {self.top_hypotheses}.

            Think step by and step and provide an analysis of their strategy, any specialization you infer from their behavior, and their competence.
            Then analyze how you can adapt your strategy to maximize efficiency and coordination with your teammate.
            Remember communication is not allowed.
            """
        else:
            user_message = f"""
            Based on the observed actions of your teammate (player_1), what do you think their strategy is?
            Are they specializing in any specific activity or subtask?

            Teammate's observed actions:
            {self.teammate_actions}

            Think step by and step and provide an analysis of their strategy, any specialization you infer from their behavior, and their competence.
            Then analyze how you can adapt your strategy to maximize efficiency and coordination with your teammate.
            Remember communication is not allowed.
            """

        return user_message

    def predict_partner_behavior(self, step, possible_teammate_strategy=None):
        if possible_teammate_strategy is None:
            possible_teammate_strategy = self.teammate_strategy
        user_message = f"""
            A dish has been delivered at step {step}.
            You previously guessed that your teammate's (player_1) policy is:  {possible_teammate_strategy}
            Based on the proposed hypothesis about your teammate (player_1), what do you think they will do next?
            Output a concise label about your teammate's next behavior in following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
            Example response:
            My teammate will place tomatoes into pot (4,2).
            ```python
            {{
              'predicted_next_behavior': 'placing tomatoes into pot (4,2)'
            }}
            """

        return user_message

    async def two_level_plan(self, state, execution_outcomes, get_action_from_response_errors, reward_tracker, step):
        hls_user_msg = self.generate_hls_user_message(state, step)
        responses = await asyncio.gather(
            *[self.controller.async_batch_prompt(self.system_message, [hls_user_msg])]
        )
        hls_response = responses[0][0] 
        self.my_strategy = deepcopy(hls_response)
        evaluator_response = ''
        subgoal_user_msg = self.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, evaluator_response, reward_tracker, step)
        responses = await asyncio.gather(
            *[self.controller.async_batch_prompt(self.system_message, [subgoal_user_msg])]
        )
        subgoal_response = responses[0][0]

        return hls_response, subgoal_response, hls_user_msg, subgoal_user_msg

    async def tom_module(self, state, execution_outcomes, get_action_from_response_errors, reward_tracker, step):
        hls_user_msg = ''
        hls_response = ''
        # step 1: evaluate hypotheses
        if self.delivery_num > 1:
            await self.eval_hypotheses(step)

        if not self.good_hypothesis_found:
            # step 2: infer teammate's strategy
            hls_user_msg1 = self.infer_teammate_strategy(step)
            hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg1
            # Query the LLM
            responses = await asyncio.gather(
                    *[self.controller.async_batch_prompt(self.system_message, [hls_user_msg1])]
                )
            response = responses[0][0]

            # Update the teammate strategy
            self.teammate_strategy = response
            self.teammate_hypotheses[self.delivery_num] = {'teammate_strategy': deepcopy(response)}
            # initialize the value of this hypothesis
            self.teammate_hypotheses[self.delivery_num]['value'] = 0.0

            # step 3: predict what partner will do next
            hls_user_msg3 = self.predict_partner_behavior(step)
            hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg3
            user_messages = [hls_user_msg3]
            # Sort the keys of self.teammate_hypotheses based on 'value', in descending order
            sorted_keys = sorted([key for key in self.teammate_hypotheses if key != self.delivery_num],
                    key=lambda x: self.teammate_hypotheses[x]['value'], 
                    reverse=True)
            # Loop through the top k keys
            for key in sorted_keys[:self.top_k]:
                # Access and use the key and its corresponding 'value'
                possible_teammate_strategy = self.teammate_hypotheses[key]
                hls_user_msg3 = self.predict_partner_behavior(step, possible_teammate_strategy)
                user_messages.append(hls_user_msg3)

            # Make sure output dict syntax is correct
            correct_syntax = False
            counter = 0
            while not correct_syntax and counter < 10:
                correct_syntax = True
                # Gathering responses asynchronously
                responses = await asyncio.gather(
                    *[self.controller.async_batch_prompt(self.system_message, [user_msg]) 
                    for user_msg in user_messages]
                    )           
                for i in range(len(responses)):
                    response = responses[i][0]
                    next_behavior = self.parse_multiple_llm_responses(response)
                    if 'predicted_next_behavior' not in next_behavior:
                        correct_syntax = False
                        print(f"Error parsing dictionary when extracting next behavior, retrying...")
                        break
                    if i == 0:
                        self.next_behavior = deepcopy(next_behavior)
                        self.teammate_hypotheses[self.delivery_num]['next_behavior'] = deepcopy(self.next_behavior)
                        # add response to hls after two new lines
                        hls_response = hls_response + '\n\n' + response
                    else:
                        self.teammate_hypotheses[sorted_keys[i-1]]['next_behavior'] = deepcopy(next_behavior)
                counter += 1
        else:
            # skip asking about opponent's strategy when we have a good hypothesis
            # Sort the keys of self.teammate_hypotheses based on 'value', in descending order
            sorted_keys = sorted([key for key in self.teammate_hypotheses], key=lambda x: self.teammate_hypotheses[x]['value'], reverse=True)
            # set the possible teammate strategy to the top hypothesis
            best_key = sorted_keys[0]
            # assert the value of the best key is above the threshold
            assert self.teammate_hypotheses[best_key]['value'] > self.good_hypothesis_thr
            self.teammate_strategy = deepcopy(self.teammate_hypotheses[best_key])
            good_hypothesis_summary = f"""Good hypothesis found: {self.teammate_strategy}"""

        # step 4: Generate your own strategy based on what teammate will do
        hls_user_msg2 = self.generate_hls_user_message(state, step)
        hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg2
        # Query the LLM
        responses = await asyncio.gather(
                        *[self.controller.async_batch_prompt(self.system_message, [hls_user_msg2])]
                    )
        hls_response = responses[0][0] 
        self.my_strategy = deepcopy(hls_response)

        return hls_response, hls_user_msg, self.teammate_strategy

    async def eval_hypotheses(self, step):
        latest_key = max(self.teammate_hypotheses.keys()) # should this be evaluated when hypothesis is good?
        # Sort the keys of self.teammate_hypotheses based on 'value', in descending order
        sorted_keys = sorted([key for key in self.teammate_hypotheses if key != latest_key],
                    key=lambda x: self.teammate_hypotheses[x]['value'], 
                    reverse=True)
        keys2eval = sorted_keys[:self.top_k] + [latest_key]
        # Loop through the top N keys and the latest key
        self.good_hypothesis_found = False
        predicted_behaviors = []
        for key in keys2eval:
            # Access and use the key and its corresponding 'value'
            if 'predicted_next_behavior' not in self.teammate_hypotheses[key]['next_behavior']:
                # set a temporary value for the next behavior
                self.teammate_hypotheses[key]['next_behavior'] = {'predicted_next_behavior': 'unknown'}
            predicted_next_behavior = self.teammate_hypotheses[key]['next_behavior']['predicted_next_behavior']
            predicted_behaviors.append(predicted_next_behavior)
            hls_user_msg3 = self.evaluate_predicted_behavior(step, predicted_next_behavior)
            user_messages.append(hls_user_msg3)

        # call LLM to ask if any of the predicted behaviors are correct, and use to eval hypotheses
        # Make sure output dict syntax is correct
        correct_syntax = False
        counter = 0
        while not correct_syntax and counter < 10:
            correct_syntax = True
            # Gathering responses asynchronously
            responses = await asyncio.gather(
                *[self.controller.async_batch_prompt(self.system_message, [user_msg])
                for user_msg in user_messages]
                )
            for i in range(len(responses)):
                key = keys2eval[i]
                response = responses[i][0]
                pred_label = self.parse_multiple_llm_responses(response)
                if 'evaluate_predicted_behavior' not in pred_label or pred_label['evaluate_predicted_behavior'] not in [True, False]:
                    correct_syntax = False
                    print(f"Error parsing dictionary when extracting label, retrying...")
                    break
                # if true, increase value of hypothesis
                if pred_label['evaluate_predicted_behavior']:
                    # update the value of this hypothesis with a Rescorla Wagner update
                    prediction_error = self.correct_guess_reward - self.teammate_hypotheses[key]['value']
                elif not pred_label['evaluate_predicted_behavior']:
                    prediction_error = -self.correct_guess_reward - self.teammate_hypotheses[key]['value']
                else:
                    # something weird happened
                    prediction_error = 0
                self.teammate_hypotheses[key]['value'] += self.alpha * prediction_error

        if self.teammate_hypotheses[key]['value'] > self.good_hypothesis_thr:
            self.good_hypothesis_found = True
                
    def evaluate_predicted_behavior(self, step, predicted_next_behavior):
        # get teammate actions since last step delivered
        latest_teammate_actions = [action for action in self.teammate_actions if action['step'] > self.last_step_delivered]
        user_message = f"""
            A dish has been delivered at step {step}.
            You previously guessed that your teammate's (player_1) would perform this behavior in this round:  {predicted_next_behavior}
            Here is the observed behavior of your teammate (player_1) in this round: {latest_teammate_actions}
            Did your prediction match the observed behavior?
            Concisely output True or False in the below Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
            Example response:
            ```python
            {{
              'evaluate_predicted_behavior': True
            }}
            """

        return user_message
    
    def parse_multiple_llm_responses(self, responses, response_type=None, state=None):
        """Parses the critic's response and returns the feedback."""
        if response_type == 'subgoal':
            for i, response in enumerate(responses):
                response_dict = self.extract_dict(response)
                if response_dict == {}:
                    continue
                elif not self.is_valid_plan(state, response_dict):
                    continue
                else:
                    goals_and_actions = response_dict
                    subgoal_response = response
                    return subgoal_response, goals_and_actions
                
            return '', {}
        elif response_type == 'next_inventories':
            for i, response in enumerate(responses):
                response_dict = self.extract_dict(response)
                if response_dict == {}:
                    continue
                elif 'predicted_opponent_next_inventory' not in response_dict:
                    continue
                elif 'my_next_inventory' not in response_dict:
                    continue
                else:
                    return response, response_dict
        else:
            for i, response in enumerate(responses):
                response_dict = self.extract_dict(response)
                if response_dict == {}:
                    continue
                else:
                    return response, response_dict
                
            return '', {}

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

    def extract_goals_and_actions(self, response, state):
        goals_and_actions = self.parse_multiple_llm_responses(response, response_type='subgoal', state=state)
        #goals_and_actions = self.extract_dict(response)
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

        # Update memory of what other agent is doing
        if step > 0:
            # check for transition of teammate putting tomato in pot
            last_ego_state = deepcopy(self.prev_ego_states[-1])
            # Check if player_0 didn't put a tomato in the pot in the last step
            player_0_interacted_with_pot = False
            player_0_pos = None
            player_0_orient = None
            for key, value in last_ego_state.items():
                if key.startswith('player_0'):
                    player_0_pos = value[0]
                    player_0_orient = key.split('-')[1]
                    player_0_key = key
                    break

            tomatoes_accounted_for = 0
            if player_0_pos and player_0_orient and 'tomato' in player_0_key:
                if (self.player_spot == 'left' and player_0_pos in [(3, 2), (3, 3)] and player_0_orient == 'E' and last_ego_state['action'] == 'INTERACT') or \
                (self.player_spot == 'right' and player_0_pos in [(5, 2), (5, 3)] and player_0_orient == 'W' and last_ego_state['action'] == 'INTERACT'):
                    player_0_interacted_with_pot = True
                    tomatoes_accounted_for = 1

            # Check for unaccounted increase in tomato numbers in the pots
            for location in [(4, 2), (4, 3)]:
                prev_num_tomatoes = self.pot_state_memory[-1][location]['num_tomatoes']
                curr_num_tomatoes = self.pot_state[location]['num_tomatoes']
                if curr_num_tomatoes - prev_num_tomatoes > tomatoes_accounted_for:
                    # Teammate put a tomato in the pot
                    print(f"Teammate put a tomato in the pot at {location}")
                    # Update memory or take any necessary action based on this information
                    # For example, you can add this information to a list or dictionary
                    # to keep track of the teammate's actions
                    teammate_action = {'action': 'put tomato in pot '+str(location), 'step': step}
                    self.teammate_actions.append(teammate_action)

            # go through self.prev_ego_states in reverse order to see what the teammate was holding in the last step (or last time we saw them)
            teammate_prev_holding = 'invisible'
            for i in range(len(self.prev_ego_states)-1, -1, -1):
                prev_ego_state = self.prev_ego_states[i]
                # if a key starts with 'player_1' in prev_ego_state 
                teammate_key = [key for key in prev_ego_state.keys() if key.startswith('player_1')]
                if len(teammate_key) > 0:
                    teammate_prev_holding = next(('tomato' if 'tomato' in key else 'empty_dish' if 'dish' in key else 'cooked_soup' if 'cooked_soup' in key else 'nothing' for key in teammate_key), 'nothing')
                    break

            teammate_key2 = [key for key in ego_state.keys() if key.startswith('player_1')]
            if len(teammate_key2) > 0:
                teammate_curr_holding = next(('tomato' if 'tomato' in key else 'empty_dish' if 'dish' in key else 'cooked_soup' if 'cooked_soup' in key else 'nothing' for key in teammate_key2), 'nothing')
            else:
                teammate_curr_holding = 'invisible'

            # check for transition of teammate picking up dish
            if teammate_curr_holding == 'invisible' or teammate_prev_holding == 'invisible':
                teammate_action = {}
            elif teammate_prev_holding == 'nothing' and teammate_curr_holding == 'empty_dish':
                print(f"Teammate picked up a dish")
                teammate_action = {'action': 'pick up dish', 'step': step}
                self.teammate_actions.append(teammate_action)
            elif teammate_prev_holding == 'empty_dish' and teammate_curr_holding == 'nothing':
                print(f"Teammate put down a dish")
                teammate_action = {'action': 'put down dish', 'step': step}
                self.teammate_actions.append(teammate_action)
            elif teammate_prev_holding == 'empty_dish' and teammate_curr_holding == 'cooked_soup':
                print(f"Teammate picked up cooked soup in dish")
                teammate_action = {'action': 'picked up cooked soup', 'step': step}
                self.teammate_actions.append(teammate_action)
            elif teammate_prev_holding == 'cooked_soup' and teammate_curr_holding == 'nothing':
                # TO DO - MAKE THIS BASED ON REWARD AND SIMILAR LOGIC TO ABOVE
                print(f"Teammate delivered cooked soup")
                teammate_action = {'action': 'delivered soup', 'step': step}
                self.teammate_actions.append(teammate_action)
            elif teammate_prev_holding == 'nothing' and teammate_curr_holding == 'tomato':
                print(f"Teammate picked up a tomato")
                teammate_action = {'action': 'picked up tomato', 'step': step}
                self.teammate_actions.append(teammate_action)
            else:
                teammate_action = {}

            if teammate_action == {}:
                # Check if teammate has been standing at the same location for more than 20 frames
                if len(self.prev_ego_states) > 20:
                    teammate_spotted_frames = []
                    for i in range(len(self.prev_ego_states) - 1, -1, -1):
                        prev_ego_state = self.prev_ego_states[i]
                        teammate_key = [key for key in prev_ego_state.keys() if key.startswith('player_1')]
                        if teammate_key:
                            teammate_location = prev_ego_state[teammate_key[0]][0]
                            teammate_spotted_frames.append((teammate_location, i))
                        if len(teammate_spotted_frames) > 1 and teammate_spotted_frames[-1][1] - teammate_spotted_frames[0][1] > 20:
                            if len(set(loc for loc, _ in teammate_spotted_frames)) == 1:
                                standing_location = teammate_spotted_frames[0][0]
                                if len(self.teammate_actions) == 0 or 'standing around at' not in self.teammate_actions[-1]['action']:
                                    print(f"Teammate has been standing around at {standing_location} for more than 20 frames")
                                    teammate_action = {'action': f'standing around at {standing_location}', 'step': step}
                                    self.teammate_actions.append(teammate_action)
                            break

        self.prev_ego_states.append(deepcopy(ego_state))

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

        # Update memory of pot state
        self.pot_state_memory.append(deepcopy(self.pot_state))

        # Update memory of what other agent is doing
        # check for transition of teammate putting tomato in pot
        if step > 0:
            self.prev_ego_states[-1]['action'] = deepcopy(action)
        # last_ego_state = self.prev_ego_states[-1]
        # # Check if player_0 didn't put a tomato in the pot in the last step
        # player_0_interacted_with_pot = False
        # player_0_pos = None
        # player_0_orient = None
        # for key, value in last_ego_state.items():
        #     if key.startswith('player_0'):
        #         player_0_pos = value[0]
        #         player_0_orient = key.split('-')[1]
        #         player_0_key = key
        #         break

        # tomatoes_accounted_for = 0
        # if player_0_pos and player_0_orient and 'tomato' in player_0_key:
        #     if (self.player_spot == 'left' and player_0_pos in [(3, 2), (3, 3)] and player_0_orient == 'E' and last_ego_state['action'] == 'INTERACT') or \
        #     (self.player_spot == 'right' and player_0_pos in [(5, 2), (5, 3)] and player_0_orient == 'W' and last_ego_state['action'] == 'INTERACT'):
        #         player_0_interacted_with_pot = True
        #         tomatoes_accounted_for = 1
        #         breakpoint()

        # # Check for unaccounted increase in tomato numbers in the pots
        # for location in [(4, 2), (4, 3)]:
        #     prev_num_tomatoes = self.pot_state_memory[-1][location]['num_tomatoes']
        #     curr_num_tomatoes = self.pot_state[location]['num_tomatoes']
        #     if curr_num_tomatoes - prev_num_tomatoes > tomatoes_accounted_for:
        #         # Teammate put a tomato in the pot
        #         print(f"Teammate put a tomato in the pot at {location}")
        #         # Update memory or take any necessary action based on this information
        #         # For example, you can add this information to a list or dictionary
        #         # to keep track of the teammate's actions
        #         teammate_action = {'action': 'put tomato in pot '+str(location), 'step': step}
        #         self.teammate_actions.append(teammate_action)
        #         breakpoint()

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