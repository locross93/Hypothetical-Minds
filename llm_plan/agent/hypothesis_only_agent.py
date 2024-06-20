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
        self.my_hypotheses = {}
        self.interaction_num = 0
        self.good_hypothesis_found = False
        self.alpha = 0.3 # learning rate for updating hypothesis values
        self.correct_guess_reward = 3.0
        self.good_hypothesis_thr = 3.0
        self.top_k = 5 # number of top hypotheses to evaluate
        self.self_improve = config['self_improve']
        player_key = self.agent_id
        opponent_key = ['player_1' if self.agent_id == 'player_0' else 'player_0'][0]
        for entity_type in ['yellow_box', 'blue_box', 'purple_box', 'ground', player_key, opponent_key]:
            self.memory_states[entity_type] = []

    def generate_system_message(self):
        system_message = f"""
            You are Agent {self.agent_id} in the two player 'running_with_scissors_in_the_matrix__repeated'
            Melting Pot multiagent reinforcement learning environment that is an 23x15 (x by y) grid with resources to collect and walls to navigate around. 
            Players can move around the map and collect resources of 3 discrete types corresponding to rock, paper, and
            scissors strategies - Yellow box = rock  - Purple box = paper - Blue box = scissors. 
            In addition to movement, the agents have an action to fire an "interaction" beam which initiates a duel, a modified version of the classic rock paper scissors game.
            Remember strategies are not transitive for rock paper scissors and the outcomes are zero-sum; here one player gets positive reward and the other agent gets an opposite negative reward according to their inventories.
            Rock/yellow beats scissors/blue (rock crushes scissors), paper/purple beats rock/yellow (paper covers rock), and scissors/blue beats paper/purple (scissors cuts paper).
            All players carry an inventory with the count of resources picked up since last respawn and for each respawn start with an inventory of 1 resource each.
            This inventory is visible in the state with the key 'inventory'.
            To play a pure strategy strongly, pick up at least 5 resources or more of the color and then fire the interaction beam at the other player.
            To commit less strongly to a strategy, pick up around 3 resources of the color and then fire the interaction beam at the other player.
            Usually you will only want to pick up one type of resource before an interaction, in order to gain the most information about the other player's strategy and to not waste time collecting resources.
            Your reward is the result of a matrix multiplication involving the your inventory in a vector format, and your opponent's inventory vector, and a payoff matrix similar to rock paper scissors.
            r_t = transpose(your_inventory) * A_payoff * opponent_inventory
            where A_payoff = np.array([[0, -10, 10], [10, 0, -10], [-10, 10, 0]])
            The reward can range from (10, -10) depending on the inventories of both players.
            For example inventories of player0_inventory = [1, 1, 10] and player1_inventory = [10, 1, 1]  (Yellow, Purple, Blue) gives Player 0: -5.625 and Player 1: 5.625 reward.
            In this example, player0 may want to play a pure paper/purple strategy in the next interaction if they anticipate that player1 will stick to their rock/yellow strategy.
            To fire the interaction beam, use the fire_at(target_coord) action function, but you may typically want to move to the middle of the environmnent first to meet your opponent.
            State Description: This environment is partially-observable, you can observed a 5x5 grid around your agent depending on your position and orientation (you can see more in front of you than behind).
            Previously seen states will be represented in memory, but note that these states could potentially be outdated. For example the other agent could collect a resource that you previously saw.
            Given the partially-observable nature of the environment, you will need to explore the environment appropriately and select goals based on the information you've gathered.
            Also pay attention to your opponent's position when you see it in order to duel with them and gain information about their strategy.

            Objective:
            Your task is to devise efficient action plans for player {self.agent_id} to play a strategy against the other player.
            First decide which strategy to play. Early on in the episode you will not know what strategy the other player is playing. But over time you will accumulate evidence about their strategy.
            Additionally you can follow the other player around to gain more information about their strategy and use other clues such as missing resources.
            Then based on your chosen strategy, you should plan a sequence of subgoals to execute this strategy such as collecting resources until you've reached your target inventory, then dueling with the other player.
            Then use the specified action functions to execute these subgoals.

            Consider the entire game state to plan the most efficient paths for resource collection and strategy execution.
            To do this you will need to think step by step about what actions to output in the following format for 
            these players to efficiently collect the appropriate resources/target inventories and play their strategy.
            Take into account the proximity of the target_coord from the src_coord and the shortest path to get to a target resource.

            ONLY USE THESE 2 ACTION FUNCTIONS:
            - move_to(src_coord, target_coord): Efficiently move agent from source coordinate to target coordinate. DO NOT move to any locations where walls are present.
            - fire_at(target_coord): Stay around specified coordinate and fire interaction when opponent is spotted to initiate duel. 
            After an interaction both players respawn.

            Keep in mind that you will be prompted again with an updated game state after these action plans are executed.
            Therefore you do not need to plan everything at once and currently can output a subgoal or series of subgoals as you wish.
            """
        self.system_message = f"As the controller for agent {self.agent_id}, focus only on strategies for this specific agent. Ignore strategies for other players. {system_message}"
    
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
            You are at step {step} of the game.
            You have decided to execute a high-level strategy/target inventory in a previous response given what you predicted your opponent will do.
            Select subgoals in order to achieve the target inventory, denoted by 'my_next_inventory' in this dictionary: {self.next_inventory}.
            Your task is to devise efficient action plans for player {self.agent_id}, reason through what the next subgoals should be given the state information. 
            Your response should be broken up into two parts:
                1. Subgoal Plan - based on the current state and the high-level strategy you previously specified above, 
                decompose this strategy into a sequence subgoals and actions to efficiently implement this strategy. Think step by step about this. This could be fairly long.
                2. Action Plan - output this sequence of actions in the following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
            Example response:
            Subgoal Plan: Given the current state, my scissors strategy to get to an inventory of 1 yellow, 1 purple, and 5 blue, and my inventory of 1 yellow, 1 purple, and 3 blue resources,
            I should move towards the two nearest observable blue box location to collect so I have 5.
            The nearest observable blue boxes are at (9, 5) and (13, 5).\n- Since I am at (11, 7), the closest one is at (9, 5).
            I should move there first and then move to (13, 5). After these two actions are completed, I can move towards the middle of the environment to initiate a duel.
            ```python
            {{
              'action_plan': ['move_to((11, 7), (9, 5))', 'move_to((9, 5), (13, 5))']
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

    def generate_interaction_feedback_user_message1(
            self, 
            total_rewards,
            step):
        rewards_str = "\n".join(f"- {player}: {reward}" for player, reward in total_rewards.items())
        # get the top N hypotheses
        sorted_keys = sorted([key for key in self.my_hypotheses],
                    key=lambda x: self.my_hypotheses[x]['value'], 
                    reverse=True)
        top_keys = sorted_keys[:self.top_k]
        # show top hypotheses with value > 0
        self.top_hypotheses = {key: self.my_hypotheses[key] for key in top_keys if self.my_hypotheses[key]['value'] > 0}
        
        if self.self_improve:
            strategy_request = f"""
                An interaction with the other player has occurred at step {step}, {self.interaction_history[-1]}.
                The total interaction history is: {self.interaction_history}.
                Strategy Request:
                Provide a strategy for player {self.agent_id}. 
                Here are stategies you previously generated: {self.top_hypotheses}.
                Outline a clear high level strategy about how to play the rest of the episode over multiple interactions. This will be passed to you later on to set a target inventory for the next interaction and also after each subsequent interaction. So keep it general, rather than specific to the next interaction.
                The strategy should specify a policy for how you would play given the interaction history. Think step by step about the interaction history and how you should play from now on in response to it.
                If your previous strategies displayed above are useful, you can iterate and refine them to get a better policy.
                In the 2nd part of your response, summarize your strategy in a concise message following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
                This summary will be shown to you in the future in order for you to select a target inventory for the next interaction.
                Example summary:
                ```python
                {{
                'my_strategy': 'Cycle through playing rock, then paper, then scissors',
                }}
                ```
                """

        user_message = f"""Total Rewards:{rewards_str}

            {strategy_request}
            """
        return user_message

    def generate_interaction_feedback_user_message2(
            self, 
            step,
            possible_strategy=None):
        
        if possible_strategy is None:
            possible_strategy = self.my_strategy
        user_message = f"""
            An interaction with the other player has occurred at step {step}, {self.interaction_history[-1]}.
            The total interaction history is: {self.interaction_history}.
            You last played: {self.interaction_history[-1]}
            You previously set your current strategy as follows: {possible_strategy}.
            Target Inventory Request:
            Provide the next inventory to play for player {self.agent_id}. 
            This response should include step by step reasoning about what inventory to play given the specified strategy and the last interaction or interaction history if applicable.
            In the 2nd part of your response, output your next inventory in following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
            Example response:
            Given my strategy to cycle through plays sequentially, and since I last played rock, I believe my next inventory should be a paper policy inventory={{'rock/yellow': 1, 'paper/purple': 5, 'scissors/blue': 1}}.
            ```python
            {{
              'my_next_inventory': {{'rock/yellow': 1, 'paper/purple': 5, 'scissors/blue': 1}}
            }}
            """
        return user_message

    async def two_level_plan(self, state, execution_outcomes, get_action_from_response_errors, reward_tracker, interaction_rewards, step, after_interaction=False):
        if after_interaction:
            hls_user_msg = ''
            hls_response = ''
            # score top hypotheses based on last interaction's possible inventory
            if self.interaction_num > 1:
                cf_reward_usr_msgs, cf_reward_response = await self.eval_hypotheses()
                hls_user_msg = hls_user_msg + '\n\n'.join(cf_reward_usr_msgs)
                hls_response = hls_response + cf_reward_response

            if not self.good_hypothesis_found:
                hls_user_msg1 = self.generate_interaction_feedback_user_message1(reward_tracker, step) 
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
                    my_strategy = self.extract_dict(response)
                    key_present = 'my_strategy' in my_strategy
                    if not key_present:
                        correct_syntax = False
                        print(f"Error parsing dictionary when extracting opponent inventory, retrying...")
                    counter += 1
                self.my_strategy = deepcopy(my_strategy)
                self.my_hypotheses[self.interaction_num] = deepcopy(my_strategy)
                # initialize the value of this hypothesis
                self.my_hypotheses[self.interaction_num]['value'] = 0
                # add response to hls after two new lines
                if len(self.top_hypotheses) > 0:
                    top_hypotheses_summary = f"""Top hypotheses: {self.top_hypotheses}"""
                else:
                    top_hypotheses_summary = ""
                hls_response = hls_response + '\n\n' + top_hypotheses_summary+ '\n\n' + response

                # output next inventory for latest hypothesis and the top k so far
                hls_user_msg2 = self.generate_interaction_feedback_user_message2(step)
                hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg2
                user_messages = [hls_user_msg2]
                system_messages = [self.system_message for i in range(3)]
                # Sort the keys of self.opponent_hypotheses based on 'value', in descending order
                sorted_keys = sorted([key for key in self.my_hypotheses if key != self.interaction_num],
                        key=lambda x: self.my_hypotheses[x]['value'], 
                        reverse=True)
                # Loop through the top k keys
                for key in sorted_keys[:self.top_k]:
                    # Access and use the key and its corresponding 'value'
                    my_strategy = self.my_hypotheses[key]
                    hls_user_msg2 = self.generate_interaction_feedback_user_message2(step, my_strategy)
                    user_messages.append(hls_user_msg2)

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
                        next_inventory = self.extract_dict(response)
                        key_present = 'my_next_inventory' in next_inventory
                        if not key_present:
                            correct_syntax = False
                            print(f"Error parsing dictionary when extracting next inventory, retrying...")
                            break
                        if i == 0:
                            self.next_inventory = deepcopy(next_inventory)
                            self.my_hypotheses[self.interaction_num]['next_inventory'] = deepcopy(self.next_inventory)
                            # add response to hls after two new lines
                            hls_response = hls_response + '\n\n' + response
                        else:
                            self.my_hypotheses[sorted_keys[i-1]]['next_inventory'] = deepcopy(next_inventory)
                    counter += 1
                subgoal_user_msg = self.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
                responses = await asyncio.gather(
                    *[self.controller.async_batch_prompt(self.system_message, [subgoal_user_msg])]
                )
                subgoal_response = responses[0][0]
            else:
                # skip asking about strategy when we have a good hypothesis
                # Sort the keys of self.my_hypotheses based on 'value', in descending order
                sorted_keys = sorted([key for key in self.my_hypotheses], key=lambda x: self.my_hypotheses[x]['value'], reverse=True)
                # set the strategy to the top hypothesis
                best_key = sorted_keys[0]
                # assert the value of the best key is above the threshold
                assert self.my_hypotheses[best_key]['value'] > self.good_hypothesis_thr
                self.my_strategy = deepcopy(self.my_hypotheses[best_key])
                good_hypothesis_summary = f"""Good hypothesis found: {self.my_strategy}"""
                # add summary to hls after two new lines
                hls_response = good_hypothesis_summary

                # output next inventory for current good strategy
                hls_user_msg2 = self.generate_interaction_feedback_user_message2(step)
                hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg2
                responses = await asyncio.gather(
                    *[self.controller.async_batch_prompt(self.system_message, [hls_user_msg2])]
                )
                response = responses[0][0]
                self.next_inventory = self.extract_dict(response)
                self.my_hypotheses[best_key]['next_inventory'] = deepcopy(self.next_inventory)
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
            self.next_inventory = deepcopy(hls_response)
            subgoal_user_msg = self.generate_feedback_user_message(state, execution_outcomes, get_action_from_response_errors, reward_tracker, step)
            responses = await asyncio.gather(
                *[self.controller.async_batch_prompt(self.system_message, [subgoal_user_msg])]
            )
            subgoal_response = responses[0][0]
            
        return hls_response, subgoal_response, hls_user_msg, subgoal_user_msg

    async def eval_hypotheses(self):
        self.good_hypothesis_found = False
        if not self.good_hypothesis_found:
            # latest hypothesis is the online one that achieved rewards
            online_key = max(self.my_hypotheses.keys())
            # Sort the keys of self.my_hypotheses based on 'value', in descending order
            sorted_keys = sorted([key for key in self.my_hypotheses if key != online_key],
                        key=lambda x: self.my_hypotheses[x]['value'], 
                        reverse=True)
            # evaluate the top k with counterfactual offline rewards
            keys2eval = sorted_keys[:self.top_k]
        else:
            # Sort the keys of self.my_hypotheses based on 'value', in descending order
            sorted_keys = sorted([key for key in self.my_hypotheses],
                        key=lambda x: self.my_hypotheses[x]['value'], 
                        reverse=True)
            # latest hypothesis is the one that was good
            online_key = sorted_keys[0]
            # evaluate the remaining top k with counterfactual offline rewards
            keys2eval = sorted_keys[1:self.top_k]

        last_round = self.interaction_history[-1]
        online_reward = last_round['rewards']
        prediction_error = online_reward - self.my_hypotheses[online_key]['value']
        self.my_hypotheses[online_key]['value'] = self.my_hypotheses[online_key]['value'] + (self.alpha * prediction_error)
        if self.my_hypotheses[online_key]['value'] > self.good_hypothesis_thr:
            self.good_hypothesis_found = True

        # Loop through the top N keys
        if len(keys2eval) == 0:
            return '', ''

        user_messages = []
        for key in keys2eval:
            hypothetical_inventory = self.my_hypotheses[key]['next_inventory']
            cf_user_msg = self.generate_counterfactual(hypothetical_inventory, last_round)
            user_messages.append(cf_user_msg)

        # Gathering responses asynchronously
        responses = await asyncio.gather(
            *[self.controller.async_batch_prompt(self.system_message, [user_msg]) 
            for user_msg in user_messages]
            )           
        for i in range(len(responses)):
            response = responses[i][0]
            cf_reward = self.extract_dict(response)
            if cf_reward['counterfactual_reward'] == 'positive':
                offline_reward = self.correct_guess_reward
            elif cf_reward['counterfactual_reward'] == 'negative':
                offline_reward = -self.correct_guess_reward
            elif cf_reward['counterfactual_reward'] == 'neutral':
                offline_reward = 0
            prediction_error = offline_reward - self.my_hypotheses[keys2eval[i]]['value']
            self.my_hypotheses[keys2eval[i]]['value'] = self.my_hypotheses[keys2eval[i]]['value'] + (self.alpha * prediction_error)

            if self.my_hypotheses[key]['value'] > self.good_hypothesis_thr:
                self.good_hypothesis_found = True

        return user_messages, ', '.join(responses[0])

    def generate_counterfactual(self, hypothetical_inventory, last_round):
        user_message = f"""
            An interaction with the other player has occurred.
            You last played: {last_round}
            Generate a counterfactual about what reward you would have received if you played the hypothetical inventory: {hypothetical_inventory} instead.
            Respond about whether you would have achieved positive, negative, or neutral (-0.5 to 0.5) reward given the reward conditions described below.
            First think about what inventory your opponent played given the last interaction and then reason about what reward you would have received if you played the hypothetical inventory.
            - if you played a strategy of rock and received a moderate or high positive reward (0.5,10], then the opponent likely played a scissors strategy
            - if you played a strategy of rock and received a moderate or large negative reward reward [-10,-0.5], then the opponent likely played a paper strategy
            - if you played a strategy of rock and received a neutral reward near zero [-0.5,-0.5], then the opponent likely also played a paper strategy
            - if you played a strategy of paper and received a moderate or high positive reward (0.5,10], then the opponent likely played a rock strategy
            - if you played a strategy of paper and received a moderate or large negative reward reward [-10,-0.5], then the opponent likely played a scissors strategy
            - if you played a strategy of paper and received a neutral reward near zero [-0.5,-0.5], then the opponent likely also played a paper strategy
            - if you played a strategy of scissors and received a moderate or high positive reward (0.5,10], then the opponent likely played a paper strategy
            - if you played a strategy of scissors and received a moderate or large negative reward reward [-10,-0.5], then the opponent likely played a rock strategy
            - if you played a strategy of scissors and received a neutral reward near zero [-0.5,-0.5], then the opponent likely also played a scissor strategy
            In the 2nd part of your response, output positive, negative, or neutral in following Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
            Example response:
            Given my strategy to cycle through plays sequentially, and since I last played rock, I believe my next inventory should be a paper policy inventory={{'rock/yellow': 1, 'paper/purple': 5, 'scissors/blue': 1}}.
            ```python
            {{
              'counterfactual_reward': 'positive'
            }}
            """
        return user_message

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