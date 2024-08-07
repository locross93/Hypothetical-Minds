import re
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
        self.interact_steps = 0
        self.interaction_history = []
        self.opponent_hypotheses = {}
        self.interaction_num = 0
        self.good_hypothesis_found = False
        self.alpha = 0.3 # learning rate for updating hypothesis values
        self.correct_guess_reward = 1
        self.good_hypothesis_thr = 0.7
        self.top_k = 5 # number of top hypotheses to evaluate
        self.self_improve = config['self_improve']
        player_key = self.agent_id
        opponent_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
        for entity_type in ['yellow_box', 'blue_box', 'purple_box', 'ground', player_key, opponent_key]:
            self.memory_states[entity_type] = []

    def generate_system_message(self):
        self.system_message = f"""
            You are Agent {self.agent_id} in the two player 'running_with_scissors_in_the_matrix__repeated'
            Melting Pot multiagent reinforcement learning environment that is an 23x15 (x by y) grid with resources to collect and walls to navigate around. 
            Players can move around the map and collect resources of 3 discrete types corresponding to rock, paper, and
            scissors strategies - Yellow box = rock  - Purple box = paper - Blue box = scissors. 
            Rock/yellow beats scissors/blue, paper/purple beats rock/yellow, and scissors/blue beats paper/purple.
            In addition to movement, the agents have an action to fire an "interaction" beam which initiates a duel 
            with one player getting positive reward and the other agent getting an opposite negative reward according to their inventories.
            All players carry an inventory with the count of resources picked up since last respawn and for each respawn start with an inventory of 1 resource each.
            This inventory is visible in the state with the key 'inventory'.
            To play a pure strategy strongly, pick up at least 5 resources or more of the color and then fire the interaction beam at the other player.
            To commit less strongly to a strategy, pick up around 3 resources of the color and then fire the interaction beam at the other player.
            Usually you will only want to pick up one type of resource before an interaction, in order to gain the most information about the other player's strategy and to not waste time collecting other resources.
            Your opponent will also almost always only pick up one type of resource before an interaction.
            For example, player0_inventory = [7, 1, 1]  (Yellow, Purple, Blue) is a good inventory that will lead to an informative duel, whereas player0_inventory = [2, 2, 2]  (Yellow, Purple, Blue) will not be informative.
            Your reward is the result of a matrix multiplication involving the your inventory in a vector format, and your opponent's inventory vector, and a payoff matrix similar to rock paper scissors.
            r_t = transpose(your_inventory) * A_payoff * opponent_inventory
            where A_payoff = np.array([[0, -10, 10], [10, 0, -10], [-10, 10, 0]])
            The reward usually ranges from (5, -5) depending on the inventories of both players (the min is -10 and max 10, but it is rare to get these magnitudes). Typically +/- 3-5 is a high magnitude, and a reward near 0 suggests both players played a similar inventory.
            For example inventories of player0_inventory = [1, 1, 10] and player1_inventory = [10, 1, 1]  (Yellow, Purple, Blue) gives Player 0: -5.625 and Player 1: 5.625 reward.
            And inventories of player0_inventory = [1, 5, 1] and player1_inventory = [5, 1, 1]  (Yellow, Purple, Blue) gives Player 0: 3.265 and Player 1: -3.265 reward.
            State Description: This environment is partially-observable, you can observed a 5x5 grid around your agent depending on your position and orientation (you can see more in front of you than behind).
            Previously seen states will be represented in memory, but note that these states could potentially be outdated. For example the other agent could collect a resource that you previously saw.
            Given the partially-observable nature of the environment, you will need to explore the environment appropriately and select goals based on the information you've gathered.
            Also pay attention to your opponent's position when you see it in order to duel with them and gain information about their strategy.
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
        yellow_locations = ego_state.get('yellow_box', [])
        blue_locations = ego_state.get('blue_box', [])
        purple_locations = ego_state.get('purple_box', [])
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
            High level strategy: I want to play a pure scissors strategy and collect 5 blue resources.
            You will be prompted again shortly to select subgoals and action plans to execute this strategy, so do not include that in your response yet right now.
            """
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

        player_position_list = next(iter(player_position.values()))
        current_position = player_position_list[0] if player_position_list else None
        if current_position is not None:
            self.update_memory_states_with_distance(player_position_list[0])

        yellow_locations = ego_state.get('yellow_box', [])
        blue_locations = ego_state.get('blue_box', [])
        purple_locations = ego_state.get('purple_box', [])
        beam_locations = ego_state.get('beam', [])
        ground_locations = ego_state.get('ground', [])
        opponent_locations = [v[0] for k, v in ego_state.items() if k.startswith('player_1' if self.agent_id == 'player_0' else 'player_0')]
        
         # Calculate the distance from the current location to the yellow, blue, and purple box locations
        yellow_locations_with_distance = [(loc, f"distance: {self.calculate_manhattan_distance(current_position, loc)}") for loc in yellow_locations]
        blue_locations_with_distance = [(loc, f"distance: {self.calculate_manhattan_distance(current_position, loc)}") for loc in blue_locations]
        purple_locations_with_distance = [(loc, f"distance: {self.calculate_manhattan_distance(current_position, loc)}") for loc in purple_locations] 

        rewards_str = "\n".join(f"- {player}: {reward}" for player, reward in rewards.items())
        
        strategy_request = f"""
            Strategy Request:
            You are at step {step} of the game.
            You have decided to execute a high-level strategy/target inventory in a previous response given what you predicted your opponent will do.
            Select subgoals in order to achieve the target inventory, denoted by 'my_next_inventory' in this dictionary: {self.next_inventories}.
            Your task is to devise efficient action plans for player {self.agent_id}, reason through what the next subgoals should be given the state information. 
            Your response should be broken up into two parts:
                1. Subgoal Plan - based on the current state and the high-level strategy you previously specified above, 
                decompose this strategy into a sequence subgoals and actions to efficiently implement this strategy. Think step by step about this. This could be fairly long.
                2. Action Plan - output this sequence of actions in the following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
            Example response 1:
            Subgoal Plan: Given the current state, my scissors strategy to get to an inventory of 1 yellow, 1 purple, and 5 blue, and my inventory of 1 yellow, 1 purple, and 3 blue resources,
            I should move towards the two nearest observable blue box location to collect so I have 5.
            The nearest observable blue boxes are at (9, 5) and (13, 5).\n- Since I am at (11, 7), the closest one is at (9, 5).
            I should move there first and then move to (13, 5). After these two actions are completed, I can move towards the middle of the environment to initiate a duel.
            ```python
            {{
              'action_plan': ['move_to((11, 7), (9, 5))', 'move_to((9, 5), (13, 5))']
            }}
            ```
            Example response 2:
            Subgoal Plan: Currently, my inventory is 1 yellow, 4 purple, and 1 blue, and I am at position (10, 10). 
            My strategy is to strengthen the paper strategy by collecting more purple resources. The nearest observable purple boxes are at (10, 9) and (8, 11). 
            I will first move to (10, 9) to collect the purple resource, then move to (8, 11) to collect another purple resource, aiming for an inventory of 1 yellow, 6 purple, and 1 blue. 
            After collecting these resources, I will look for the opponent to initiate a duel.
            ```python
            {{
            'action_plan': ['move_to((10, 10), (10, 9))', 'move_to((12, 9), (8, 11))']
            }}
            ```
            Example response 3:
            Subgoal Plan: I start with an inventory of 3 yellow, 1 purple, and 1 blue, positioned at (8, 5). My strategy is to adopt a rock strategy by collecting more yellow resources. 
            The nearest observable yellow boxes are at (7, 4) and (3, 6). I plan to move to (7, 4) first to pick up the yellow resource, then proceed to (3, 6) to collect another yellow resource, 
            targeting an inventory of 5 yellow, 1 purple, and 1 blue. Once I have collected these resources, I will attempt to find and duel with the opponent.
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

            Keep plans relatively short (<6 actions), especially at the early steps of an episode. You will be prompted again when the action plans are finished and more information is observed.
            """

        user_message = f"""Current State Description:
            - Global Map Size: {map_size} grid (Walls are located at the boundaries of the map and in other places that are invalid for move_to).
            - Valid Locations for move_to: {movable_locations}
            - Player Position: {player_position}
            - Player Orientation: {player_orientation}
            - Player Inventory (yellow, purple, blue): {player_inventory}
            - Egocentric Observations Size: 5x5 grid around your agent. You currently can observe the following based on your position and orientation:
            - Observable Yellow Box Locations (format: ((x,y), distance from current location): {yellow_locations_with_distance}
            - Observable Blue Box Locations: {blue_locations_with_distance}
            - Observable Purple Box Locations: {purple_locations_with_distance}
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

    def generate_interaction_feedback_user_message1(
            self, 
            step):
        
        user_message = f"""
            An interaction with the other player has occurred at step {step}, {self.interaction_history[-1]}.
            What was my opponent's likely inventory in the last round given the inventory I played and the reward received.
            Think step by step about this. First think about what resource you had the most of in your inventory, 
            and then think about which resource would beat that if you received a negative reward of -1 or worse or which resource would lose to yours if you received a positive reward of 1 or more.
            If you received a small magnitude reward near 0 and in between (-1, 1), then your opponent may have played a similar inventory to you.
            Then depending on the magnitude of the reward and the number of resources you played, you can infer the opponent's inventory and whether they played that strategy strongly (5+ of that resource) or weakly (~3 of that resource).
            An inventory of {{'rock/yellow': 1, 'paper/purple': 1, 'scissors/blue': 1}} is not possible because you need at least 2 resources of a type to play a duel.
            Here are some example interactions to help you reason about how the reward function works:
            'your_inventory': {{'rock/yellow': 3, 'paper/purple': 1, 'scissors/blue': 1}}, 'rewards': -2.285, 'possible_opponent_inventory': {{'rock/yellow': 1, 'paper/purple': 5, 'scissors/blue': 1}}
            'your_inventory': {{'rock/yellow': 5, 'paper/purple': 1, 'scissors/blue': 1}}, 'rewards': 3.571, 'possible_opponent_inventory': {{'rock/yellow': 1, 'paper/purple': 1, 'scissors/blue': 6}}
            'your_inventory': {{'rock/yellow': 1, 'paper/purple': 4, 'scissors/blue': 1}}, 'rewards': 2.0, 'possible_opponent_inventory': {{'rock/yellow': 3, 'paper/purple': 1, 'scissors/blue': 1}}
            'your_inventory': {{'rock/yellow': 1, 'paper/purple': 6, 'scissors/blue': 2}}, 'rewards': 0.555, 'possible_opponent_inventory': {{'rock/yellow': 1, 'paper/purple': 4, 'scissors/blue': 1}}
            'your_inventory': {{'rock/yellow': 1, 'paper/purple': 1, 'scissors/blue': 5}}, 'rewards': 3.265, 'possible_opponent_inventory': {{'rock/yellow': 1, 'paper/purple': 5, 'scissors/blue': 1}}
            'your_inventory': {{'rock/yellow': 1, 'paper/purple': 1, 'scissors/blue': 8}}, 'rewards': -4.666, 'possible_opponent_inventory': {{'rock/yellow': 7, 'paper/purple': 1, 'scissors/blue': 1}}
            In the 2nd part of your response, output the predicted opponent's inventory in following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
            Example output:
            Given that I last played a strong paper strategy with an inventory of {{'rock/yellow': 1, 'paper/purple': 5, 'scissors/blue': 1}} and received a reward of -3.428, I believe my opponent played a strong scissors strategy. 
            The reward suggests that my paper was beaten by their scissors, which means their inventory likely had a higher count of blue/scissors resources. 
            A possible inventory for them could be {{'rock/yellow': 1, 'paper/purple': 1, 'scissors/blue': 5}} or a similar distribution favoring scissors.
            ```python
            {{
              'possible_opponent_inventory': {{'rock/yellow': 1, 'paper/purple': 1, 'scissors/blue': 5}}
            }}
            ```
            """

        return user_message

    def generate_interaction_feedback_user_message2(
            self, 
            total_rewards,
            step):
        rewards_str = "\n".join(f"- {player}: {reward}" for player, reward in total_rewards.items())
        # get the top N hypotheses
        sorted_keys = sorted([key for key in self.opponent_hypotheses],
                    key=lambda x: self.opponent_hypotheses[x]['value'], 
                    reverse=True)
        top_keys = sorted_keys[:self.top_k]
        # show top hypotheses with value > 0
        self.top_hypotheses = {key: self.opponent_hypotheses[key] for key in top_keys if self.opponent_hypotheses[key]['value'] > 0}
        
        if self.self_improve:
            strategy_request = f"""
                An interaction with the other player has occurred at step {step}, {self.interaction_history[-1]}.
                The total interaction history is: {self.interaction_history}.
                Here are your previous hypotheses about the algorithm your opponent is playing: {self.top_hypotheses}.
                What is your opponent's likely policy given the inventories and the reward function? Think step by step about this given the interaction history. 
                If your previous hypotheses are useful, you can iterate and refine them to get a better explanation of the data observed so far.
                If a hypothesis already explains the data very well, then repeat the hypothesis in this response.
                They may be playing the same pure policy every time, a complex strategy to counter you, or anything in between. 
                They are not necessarily a smart agent that adapts to your strategy, you are just playing an algorithm. 
                Are you getting high positive or negative reward when playing the same type of inventory? 
                For example getting high positive reward every time you play many paper resources. 
                If so, your opponent may be playing a pure strategy and you can exploit this by playing the counter strategy.
                Once you have output a hypothesis about your opponent's strategy with step by step reaasoning, you can use hypothesis to inform your strategy. 
                In the 2nd part of your response, summarize your hypothesis in a concise message following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
                This summary will be shown to you in the future in order for you to select lower-level actions to implement the appropriate counter strategy.
                Example summary:
                ```python
                {{
                'Opponent_strategy': 'I think my opponent is always playing a pure scissors strategy and collecting around 5 blue resources.'
                }}
                ```
                
                You will be prompted again shortly to select subgoals and action plans to execute this strategy that achieves the target inventory, so do not include that in your response yet right now.
                """
        else:
            strategy_request = f"""
                An interaction with the other player has occurred at step {step}, {self.interaction_history[-1]}.
                The total interaction history is: {self.interaction_history}.
                What is your opponent's likely policy given the inventories and the reward function? Think step by step about this given the interaction history. 
                They may be playing the same pure policy every time, a complex strategy to counter you, or anything in between. 
                They are not necessarily a smart agent that adapts to your strategy. 
                Are you getting high positive or negative reward when playing the same type of inventory? 
                For example getting high positive reward every time you play many paper resources. 
                If so, your opponent may be playing a pure strategy and you can exploit this by playing the counter strategy.
                Once you have output a hypothesis about your opponent's strategy with step by step reaasoning, you can use hypothesis to inform your strategy. 
                In the 2nd part of your response, summarize your hypothesis in a concise message following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
                This summary will be shown to you in the future in order for you to select lower-level actions to implement the appropriate counter strategy.
                Example summary:
                ```python
                {{
                'Opponent_strategy': 'I think my opponent is always playing a pure scissors strategy and collecting around 5 blue resources.'
                }}
                ```
                
                You will be prompted again shortly to select subgoals and action plans to execute this strategy that achieves the target inventory, so do not include that in your response yet right now.
                """

        user_message = f"""Total Rewards:{rewards_str}

            {strategy_request}
            """
        return user_message

    def generate_interaction_feedback_user_message3(
            self, 
            step,
            possible_opponent_strategy=None):
        
        if possible_opponent_strategy is None:
            possible_opponent_strategy = self.possible_opponent_strategy
        user_message = f"""
            An interaction with the other player has occurred at step {step}, {self.interaction_history[-1]}.
            The total interaction history is: {self.interaction_history}.
            You last played: {self.interaction_history[-1]['your_inventory']}
            You previously guessed that their policy or strategy is: {possible_opponent_strategy}.
            High-level strategy Request:
            Provide the next high-level strategy for player {self.agent_id}. 
            This response should include step by step reasoning in parts 1 and 2 about which strategy to select based on the entire interaction history in the following format:
            1. 'Opponent_next_inventory': Given the above mentioned guess about the opponent's policy/strategy, and the last inventory you played (if their strategy is adaptive, it may not be), what is their likely inventory in the next round.
            2. 'My_next_inventory': Given the opponent's likely inventory in the next round, what should your next inventory be to counter this?
            3. In the 3rd part of your response, output the predicted opponent's next inventory and your next inventory in following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
            Example response:
            1. 'Opponent_next_inventory': Given that my opponent is playing a rock policy, I believe their next inventory will be inventory={{'rock/yellow': 5, 'paper/purple': 1, 'scissors/blue': 1}}.
            2. 'My_next_inventory': Given that my opponent is playing a rock policy, I believe my next inventory should be a paper policy inventory={{'rock/yellow': 1, 'paper/purple': 5, 'scissors/blue': 1}}.
            ```python
            {{
              'predicted_opponent_next_inventory': {{'rock/yellow': 5, 'paper/purple': 1, 'scissors/blue': 1}}
              'my_next_inventory': {{'rock/yellow': 1, 'paper/purple': 5, 'scissors/blue': 1}}
            }}
            """

        return user_message

    async def two_level_plan(self, state, execution_outcomes, get_action_from_response_errors, reward_tracker, step, after_interaction=False):
        if after_interaction:
            hls_user_msg = ''
            hls_response = ''
            hls_user_msg1 = self.generate_interaction_feedback_user_message1(step) 
            hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg1
            # Make sure output dict syntax is correct
            correct_syntax = False
            counter = 0
            while not correct_syntax and counter < 6:
                correct_syntax = True
                responses = await asyncio.gather(
                    *[self.controller.async_batch_prompt(self.system_message, [hls_user_msg1])]
                )
                response = responses[0][0] 
                possible_opponent_inventory = self.extract_dict(response)
                key_present = 'possible_opponent_inventory' in possible_opponent_inventory
                if not key_present:
                    correct_syntax = False
                    print(f"Error parsing dictionary when extracting opponent inventory, retrying...")
                counter += 1
            self.possible_opponent_inventory = deepcopy(possible_opponent_inventory)
            # add response to hls after two new lines
            hls_response = hls_response + '\n\n' + response
            # Update last interaction_dict with the contents of possible_opponent_inventory
            self.interaction_history[-1].update(self.possible_opponent_inventory)
            # score top hypotheses based on last interaction's possible inventory
            if self.interaction_num > 1:
                self.eval_hypotheses()
            if not self.good_hypothesis_found:
                hls_user_msg2 = self.generate_interaction_feedback_user_message2(reward_tracker, step) 
                hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg2
                responses = await asyncio.gather(
                    *[self.controller.async_batch_prompt(self.system_message, [hls_user_msg2])]
                )
                response = responses[0][0]
                possible_opponent_strategy = self.extract_dict(response)
                self.possible_opponent_strategy = deepcopy(possible_opponent_strategy)
                self.opponent_hypotheses[self.interaction_num] = deepcopy(possible_opponent_strategy)
                # initialize the value of this hypothesis
                self.opponent_hypotheses[self.interaction_num]['value'] = 0
                # add response to hls after two new lines
                top_hypotheses_summary = f"""Top hypotheses: {self.top_hypotheses}"""
                hls_response = hls_response + '\n\n' + top_hypotheses_summary+ '\n\n' + response

                # predict next opponent inventory for latest hypothesis and the top k so far
                hls_user_msg3 = self.generate_interaction_feedback_user_message3(step)
                hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg3
                user_messages = [hls_user_msg3]
                # Sort the keys of self.opponent_hypotheses based on 'value', in descending order
                sorted_keys = sorted([key for key in self.opponent_hypotheses if key != self.interaction_num],
                        key=lambda x: self.opponent_hypotheses[x]['value'], 
                        reverse=True)
                # Loop through the top k keys
                for key in sorted_keys[:self.top_k]:
                    # Access and use the key and its corresponding 'value'
                    possible_opponent_strategy = self.opponent_hypotheses[key]
                    hls_user_msg3 = self.generate_interaction_feedback_user_message3(step, possible_opponent_strategy)
                    user_messages.append(hls_user_msg3)

                # Make sure output dict syntax is correct
                correct_syntax = False
                counter = 0
                while not correct_syntax and counter < 6:
                    correct_syntax = True
                    # Gathering responses asynchronously
                    responses = await asyncio.gather(
                        *[self.controller.async_batch_prompt(self.system_message, [user_msg]) 
                        for user_msg in user_messages]
                        )           
                    for i in range(len(responses)):
                        response = responses[i][0]
                        next_inventories = self.extract_dict(response)
                        both_keys_present = ('predicted_opponent_next_inventory' in next_inventories) and ('my_next_inventory' in next_inventories)
                        if not both_keys_present:
                            correct_syntax = False
                            print(f"Error parsing dictionary when extracting next inventories, retrying...")
                            break
                        if i == 0:
                            self.next_inventories = deepcopy(next_inventories)
                            self.opponent_hypotheses[self.interaction_num]['next_inventories'] = deepcopy(self.next_inventories)
                            # add response to hls after two new lines
                            hls_response = hls_response + '\n\n' + response
                        else:
                            self.opponent_hypotheses[sorted_keys[i-1]]['next_inventories'] = deepcopy(next_inventories)
                    counter += 1
                # add response to hls after two new lines
                hls_response = hls_response + '\n\n' + response
                subgoal_user_msg = self.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
                responses = await asyncio.gather(
                    *[self.controller.async_batch_prompt(self.system_message, [subgoal_user_msg])]
                )
                subgoal_response = responses[0][0]
            else:
                # skip asking about opponent's strategy when we have a good hypothesis
                # Sort the keys of self.opponent_hypotheses based on 'value', in descending order
                sorted_keys = sorted([key for key in self.opponent_hypotheses], key=lambda x: self.opponent_hypotheses[x]['value'], reverse=True)
                # set the possible opponent strategy to the top hypothesis
                best_key = sorted_keys[0]
                # assert the value of the best key is above the threshold
                assert self.opponent_hypotheses[best_key]['value'] > self.good_hypothesis_thr
                self.possible_opponent_strategy = deepcopy(self.opponent_hypotheses[best_key])
                good_hypothesis_summary = f"""Good hypothesis found: {self.possible_opponent_strategy}"""
                # add summary to hls after two new lines
                hls_response = hls_response + '\n\n' + good_hypothesis_summary
                hls_user_msg3 = self.generate_interaction_feedback_user_message3(step)
                hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg3
                responses = await asyncio.gather(
                    *[self.controller.async_batch_prompt(self.system_message, [hls_user_msg3])]
                )
                response = responses[0][0]
                self.next_inventories = self.extract_dict(response)
                self.opponent_hypotheses[best_key]['next_inventories'] = deepcopy(self.next_inventories)
                # add response to hls after two new lines
                hls_response = hls_response + '\n\n' + response
                subgoal_user_msg = self.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
                responses = await asyncio.gather(
                    *[self.controller.async_batch_prompt(self.system_message, [subgoal_user_msg])]
                )
                subgoal_response = responses[0][0]    
        else:
            hls_user_msg = self.generate_hls_user_message(state, step)
            responses = await asyncio.gather(
                *[self.controller.async_batch_prompt(self.system_message, [hls_user_msg])]
            )
            hls_response = responses[0][0] 
            self.next_inventories = deepcopy(hls_response)
            subgoal_user_msg = self.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
            responses = await asyncio.gather(
                *[self.controller.async_batch_prompt(self.system_message, [subgoal_user_msg])]
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

        #breakpoint()

        return hls_response, subgoal_response, hls_user_msg, subgoal_user_msg

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
        while not valid_plan and counter < 15:
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

        if 'action_plan' not in goal_and_plan:
            print(f"Error: action_plan not found in goal_and_plan.")
            breakpoint()
        # set which subgoal we are on
        goal_and_plan['subgoal_num'] = 0 

        return user_message, subgoal_response, goal_and_plan

    def eval_hypotheses(self):
        latest_key = max(self.opponent_hypotheses.keys()) # should this be evaluated when hypothesis is good?
        # Sort the keys of self.opponent_hypotheses based on 'value', in descending order
        sorted_keys = sorted([key for key in self.opponent_hypotheses if key != latest_key],
                    key=lambda x: self.opponent_hypotheses[x]['value'], 
                    reverse=True)
        keys2eval = sorted_keys[:self.top_k] + [latest_key]
        # Loop through the top N keys and the latest key
        self.good_hypothesis_found = False
        for key in keys2eval:
            # Access and use the key and its corresponding 'value'
            if 'predicted_opponent_next_inventory' not in self.opponent_hypotheses[key]['next_inventories']:
                breakpoint()
            predicted_opponent_next_inventory = self.opponent_hypotheses[key]['next_inventories']['predicted_opponent_next_inventory']
            empirical_opp_inventory = self.interaction_history[-1]['possible_opponent_inventory']

            # Find the key with max value in each inventory
            # first check for ties in each inventory
            both_inventories = [predicted_opponent_next_inventory, empirical_opp_inventory]
            for inventory in both_inventories:
                max_value = max(inventory.values())
                max_items = [item for item, value in inventory.items() if value == max_value]
                is_tie = len(max_items) > 1
                if is_tie:
                    break

            if is_tie:
                # do not update value either way with ambiguous data
                continue

            # Find the key with max value in each inventory
            max_pred_key = max(predicted_opponent_next_inventory, key=predicted_opponent_next_inventory.get)
            max_empirical_key = max(empirical_opp_inventory, key=empirical_opp_inventory.get)
            # Check if the max item is the same in both inventories
            same_max_item = max_pred_key == max_empirical_key
            if same_max_item:
                # update the value of this hypothesis with a Rescorla Wagner update
                prediction_error = self.correct_guess_reward - self.opponent_hypotheses[key]['value']
            else:
                prediction_error = -self.correct_guess_reward - self.opponent_hypotheses[key]['value']
            self.opponent_hypotheses[key]['value'] = self.opponent_hypotheses[key]['value'] + (self.alpha * prediction_error)

            if self.opponent_hypotheses[key]['value'] > self.good_hypothesis_thr:
                self.good_hypothesis_found = True

    def extract_dict(self, response):
        try:
            # Find the start of the dictionary by looking for the "```python\n" or "\n```" delimiter
            start_marker = "```python\n"
            end_marker = "\n```"
            start_index = response.find(start_marker)
            if start_index != -1:
                start_index += len(start_marker)
            else:
                # If "```python\n" is not found, look for "\n```"
                start_marker = "\n```\n"
                end_marker = "```"
                start_index = response.find(start_marker)
                if start_index != -1:
                    start_index += len(start_marker)
                else:
                    raise ValueError("Python dictionary markers not found in GPT-4's response.")
            
            end_index = response.find(end_marker, start_index)
            if end_index == -1:
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
            extracted_dict = ast.literal_eval(cleaned_dict_str)
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
        func = getattr(action_funcs, func_name)            
        if func_name == 'move_to':                
            start, goal = func_args                  
            paths, actions, current_orient, path_found = func(start, goal, grid, self.orientation)
            if not path_found:
                print(f"No path found for action plan: {action_plan}. Making less strict action sequence")
                self.combine_all_known_states(state) # update agent.all_known_states
                goal_type = None
                for key, coordinates in self.all_known_states.items():
                    if goal in coordinates:
                        goal_type = key
                opponent_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
                labels = ['wall', 'yellow_box', 'blue_box', 'purple_box', opponent_key]
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
            self.orientation = deepcopy(current_orient)
            self.destination = deepcopy(goal)
        elif func_name == 'fire_at':
            actions = ['INTERACT_'+str(func_args)]
            # target = func_args
            # actions, current_orient = func(self.pos, self.orientation, target)
            # self.orientation = current_orient

        for action in actions:                      
            self.all_actions.put(action)

    def build_grid_from_states(self, states: Dict[str, List[Tuple[int, int]]], labels: List[str], ignore: List[Tuple[int, int]] = None) -> np.ndarray:
        """Build a grid from a dictionary of states. Setting walls and other obstacles you pass in (as str in labels) to 1, 
        and all other things as 0."""
        grid_width = 23
        grid_height = 15
        grid = np.zeros((grid_width, grid_height))
        for label, coords in states.items():
            if label == 'inventory':
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
                            (loc, step_str, distance) for loc, step_str, distance in self.memory_states[other_entity] if loc != location
                        ]

                # Update or add the location with the latest step number and remove older references of the same location
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
            labels = ['wall', 'yellow_box', 'blue_box', 'purple_box', opponent_key]
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
        self.all_known_states['wall'] = set(state['global']['wall'])

        # Add information from ego_state
        for key, coords in ego_state.items():
            if key != 'inventory' and key != 'wall':  # Exclude non-spatial data if needed
                self.all_known_states[key] = set(coords)

        # Add information from agent.memory_states
        for key, coords_with_step in self.memory_states.items():
            if key not in self.all_known_states:
                self.all_known_states[key] = set()
            for coord, _, _  in coords_with_step:
                self.all_known_states[key].add(coord)

        # Convert sets back to lists
        for key in self.all_known_states:
            self.all_known_states[key] = list(self.all_known_states[key])