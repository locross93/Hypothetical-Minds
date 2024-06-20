import ast
import openai


class GPTController:
    def __init__(self, api_key):        
        self.client = openai.Client(api_key=api_key)
        self.system_message = """
            You are a centralized controller in the 'running_with_scissors_in_the_matrix__repeated' 
            Melting Pot multiagent reinforcement learning environment that is an 18x24 grid with resources to collect and walls to navigate around. 
            Players can move around the map and collect resources of 3 discrete types corresponding to rock, paper, and
            scissors pure strategies - Blue box = scissors - Purple box = paper - Yellow box = rock. 
            In addition to movement, the agents have an action to fire an "interaction" beam which initiates a duel 
            with one player getting positive reward and the other agent getting an opposite negative reward according to their inventories.
            All players carry an inventory with the count of resources picked up since last respawn and for each respawn start with an inventory of 1 resource each.
            This inventory is visible in the state with the keys for each player's inventory being 'player_N_inventory'.
            To play a pure strategy, pick up about 5 resources of the color and then fire the interaction beam at the other player.
            To fire the interaction beam, use the fire_at(target_coord) action function, but you may typically want to move to the middle of the environmnent first to meet them.

            Objective:
            Your task is to devise efficient strategies for player 0 (scissors strategy, blue resources) and player 1 (paper strategy, purple resources). 
            Consider the entire game state to plan the most efficient paths for resource collection and strategy execution.
            To do this you will need to think step by step about what actions to output in the following format for 
            these players to efficiently collect the appropriate resources/target inventories and play their strategy.
            Take into account the proxity of the target_coord from the src_coord and the shortest path to get to a target resource.
            Your response should be broken up into two parts:
                1. Reflection - based on the current state, what is your strategy or goal for each player, and how can this strategy be
                decomposed into a sequence of actions to efficiently achieve this goal? Think step by step about this. This could be fairly long.
                2. Action Plan - what are the specific actions you would output for each player to execute this strategy?
                Output actions should be in the following format:

            Action Functions:
            - move_to(src_coord, target_coord): Efficiently move agent from source coordinate to target coordinate.
            - fire_at(target_coord): Fire interaction at a specified coordinate to initiate duel. 
            After and interaction both players respawn.

            You should also specify which one of these potential goals your action plan corresponds to in a format that will be specified later.
            Mid-level goals:
            - Collect yellow resource
            - Collect blue resource
            - Collect purple resource
            - Seek out and duel with other player

            Keep in mind that you will be prompted again with an updated game state after these action plans are executed.
            Therefore you do not need to plan everything at once and currently can output a subgoal or series of subgoals as you wish.
            """
        
    def generate_initial_user_message(self, state):
        # Extracting information from the state
        map_size = "18x24"  # Assuming the map size is constant
        player_positions = {k: v for k, v in state.items() if k.startswith('player')}
        yellow_locations = state.get('yellow_box', [])
        blue_locations = state.get('blue_box', [])
        purple_locations = state.get('purple_box', [])
        beam_locations = state.get('beam', [])
        ground_locations = state.get('ground', [])
        strategy_request = """
            Strategy Request:
            Provide a strategy for each player in Python dictionary format, parsable by `ast.literal_eval()`. 
            The strategy should be efficient, considering the shortest paths and strategic positioning for duels. 
            Format the dictionary with 'player_0' and 'player_1', detailing their strategies and action plans.
            Do not use JSON or any other formatting. The dictionary should start \
            with 'player_0' and include both players, listing their strategies \
            and action plans. For example:

            {
                'player_0': {
                    'goal': 'Collect blue resource',
                    'action_plans': ['move_to((x1, y1), (x2, y2))']
                },
                'player_1': {
                    'goal': 'Collect purple resource',
                    'action_plans': ['move_to((x1, y1), (x2, y2))']
                },
                ...
            }

            Actions should align with the mid-level goals and action functions, \
            emphasizing efficient pathfinding and playing the corresponding strategies.
            """
        user_message = f"""Current State Description:
            - Map Size: {map_size} grid (Walls are located at the boundaries of the map).
            - Player Positions and Orientations: {player_positions}
            - Beam Locations: {beam_locations if 'beam' in state else []}
            - Yellow Box Locations: {yellow_locations if 'yellow_box' in state else []}
            - Blue Box Locations: {blue_locations if 'blue_box' in state else []}
            - Purple Box Locations: {purple_locations if 'purple_box' in state else []}
            - Ground Locations: {ground_locations if 'ground' in state else []}
            
            {strategy_request}
            """
        return user_message
    
    def generate_feedback_user_message(self, state, execution_outcomes, rewards):
        # Extracting information from the state
        map_size = "18x24"  # Assuming the map size is constant
        player_positions = {k: v for k, v in state.items() if k.startswith('player')}
        yellow_locations = state.get('yellow_box', [])
        blue_locations = state.get('blue_box', [])
        purple_locations = state.get('purple_box', [])
        beam_locations = state.get('beam', [])
        ground_locations = state.get('ground', [])

        # Preparing execution outcomes and rewards sections
        execution_outcomes_str = "\n".join(f"- {player}: {outcome}" for player, outcome in execution_outcomes.items())
        rewards_str = "\n".join(f"- {player}: {reward}" for player, reward in rewards.items())

        strategy_request = """
            Strategy Request:
            Based on the current state and feedback, provide a revised strategy for each player in Python dictionary format. 
            It's crucial to use the current positions of the players as the starting points for the new action plans. 
            The objective remains the same while considering the players' updated locations.
            Ensure it can be parsed by the `ast.literal_eval()` method and aligns with the mid-level goals and action functions.
            For example:

            {
                'player_0': {
                    'goal': 'Collect blue resource',
                    'action_plans': ['move_to((x1, y1), (x2, y2))']
                },
                'player_1': {
                    'goal': 'Collect purple resource',
                    'action_plans': ['move_to((x1, y1), (x2, y2))']
                },
                ...
            }
            The objective is to have player 0 play a pure scissors strategy (blue) and player 1 play a pure paper strategy (purple).
            """

        user_message = f"""Current State Description:
            - Map Size: {map_size} grid (Walls are located at the boundaries of the map).
            - Player Positions and Orientations: {player_positions}
            - Beam Locations: {beam_locations if 'beam' in state else []}
            - Yellow Box Locations: {yellow_locations if 'yellow_box' in state else []}
            - Blue Box Locations: {blue_locations if 'blue_box' in state else []}
            - Purple Box Locations: {purple_locations if 'purple_box' in state else []}
            - Ground Locations: {ground_locations if 'ground' in state else []}

            Execution Outcomes:
            {execution_outcomes_str}

            Error for extracting and executing actions from the response:
            {get_action_from_response_errors}

            Rewards:
            {rewards_str}

            {strategy_request}
            """
        return user_message


    def get_actions_manually(self, state, prompt):
        player0_action = self.move_to_closest_color_box(state, player_num='0', color='blue')
        player1_action = self.move_to_closest_color_box(state, player_num='1', color='purple')
        goals_and_actions = {
            'player_0': {
                'goal': 'Collect blue resource',
                'action_plans': [player0_action]
            },
            'player_1': {
                'goal': 'Collect purple resource',
                'action_plans': [player1_action]
            },
        }
        response_data = {'usage': 0.0}

        return goals_and_actions, response_data


    def find_player_position(self, state, player_num):
        # Check for player_N in all four cardinal directions
        for direction in ['E', 'W', 'N', 'S']:
            key = f'player_{player_num}-{direction}'
            if key in state:
                return state[key][0]
        return None

    def find_closest_color_box(self, state, player_position, color):
        # Extract color box positions
        color_boxes = state[color+'_box']

        # Find the closest box
        closest_box = None
        min_distance = float('inf')
        for box in color_boxes:
            distance = abs(player_position[0] - box[0]) + abs(player_position[1] - box[1])
            if distance < min_distance:
                min_distance = distance
                closest_box = box

        return closest_box

    def move_to_closest_color_box(self, state, player_num, color):
        player_position = self.find_player_position(state, player_num)
        if not player_position:
            return "Player position not found"

        target_box = self.find_closest_color_box(state, player_position, color)

        if target_box:
            return f"move_to({player_position}, {target_box})"
        else:
            return "No box found"


    def get_actions_from_gpt4(self, prompt):
        full_prompt = [
            {
                "role": "system",
                "content": self.system_message,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
        try:            
            response = self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=full_prompt,
                max_tokens=3000,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            usage = {
                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens,                   
            }
            response_data = {
                'content': content,
                'usage': usage,
                }            
            goals_and_actions = self.extract_goals_and_actions(content)
            return goals_and_actions, response_data
        except Exception as e:
            print(f"Error in GPT-4 request: {e}")
            return {}, {}

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
            dict_str = response[start_index:end_index].strip()            
            # Remove the common string escape sequences for new lines and tabs
            dict_str = dict_str.replace("\\n", "").replace("\\t", "")            
            # Convert the string representation of a dictionary into an actual dictionary
            goals_and_actions = ast.literal_eval(dict_str)            
            return goals_and_actions
        except Exception as e:
            print(f"Error parsing goals and actions: {e}")
            return {}