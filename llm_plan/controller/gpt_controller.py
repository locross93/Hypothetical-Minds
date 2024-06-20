import ast
import openai


class GPTController:
    def __init__(self, api_key):        
        self.client = openai.Client(api_key=api_key)
        self.system_message = """
            You are a centralized controller in the 'commons_harvest__open' environment. 
            Your task is to analyze the game state, make strategic decisions, and 
            guide agents to maximize collective reward.

            Mid-Level Goals:
            - Harvest Apples
            - Sustain Apple Patches
            - Explore the Map (for partially observable settings)
            - Avoid Depletion
            - Evade Other Agents
            - Attack Other Agents
            - Establish a Zone  (cooperation with other agents)
            - Defend Resources

            Action Functions:
            - move_to(src_coord, target_coord): Move agent from source coordinate to target coordinate.
            - fire_at(target_coord): Fire at a specified coordinate to attack other agents or defend resources. 
            An agent hit by the beam is temporarily taken out of the game for 5 steps.

            Objective:
            Devise strategies that utilize the above mid-level goals and action \
            functions to achieve the highest possible collective reward. Strategies should balance resource management with defensive and offensive tactics, considering the temporary removal of agents hit by the beam.
            """
        
    def generate_initial_user_message(self, state):
        # Extracting information from the state
        map_size = "18x24"  # Assuming the map size is constant
        player_positions = {k: v for k, v in state.items() if k.startswith('player')}
        apple_locations = state.get('apple', [])
        beam_locations = state.get('beam', [])
        grass_locations = state.get('ground', [])  # Assuming 'ground' key refers to grass
        strategy_request = """
            Strategy Request:
            Based on the current state, provide a strategy for each player in Python \
            dictionary format and make sure it can be parsed by the `ast.literal_eval()` \
            method. 
            Do not use JSON or any other formatting. The dictionary should start \
            with 'player_0' and include all players, listing their strategies \
            and action plans. For example:

            {
                'player_0': {
                    'goal': 'Harvest Apple',
                    'action_plans': ['move_to((x1, y1), (x2, y2))', 'move_to((x2, y2), (x3, y3))', ...]
                },
                'player_1': {
                    'goal': 'Attack player_2',
                    'action_plans': ['move_to((x1, y1), (x2, y2))', 'fire_at((x2, y2))']
                },
                ...
            }

            Please list the strategies and actions as you would define them in \
            a Python script, ensuring they align with the mid-level goals and \
            action functions previously mentioned.
            """
        user_message = f"""Current State Description:
            - Map Size: {map_size} grid (Walls are located at the boundaries of the map).
            - Player Positions and Orientations: {player_positions}
            - Apple Locations: {apple_locations if 'apple' in state else []}
            - Beam Locations: {beam_locations if 'beam' in state else []}
            - Grass Locations: {grass_locations if 'grass' in state else []}
            
            {strategy_request}
            """
        return user_message
    
    def generate_feedback_user_message(self, state, execution_outcomes, rewards):
        # Extracting information from the state
        map_size = "18x24"  # Assuming the map size is constant
        player_positions = {k: v for k, v in state.items() if k.startswith('player')}
        apple_locations = state.get('apple', [])
        beam_locations = state.get('beam', [])
        grass_locations = state.get('ground', [])  # Assuming 'ground' key refers to grass

        # Preparing execution outcomes and rewards sections
        execution_outcomes_str = "\n".join(f"- {player}: {outcome}" for player, outcome in execution_outcomes.items())
        rewards_str = "\n".join(f"- {player}: {reward}" for player, reward in rewards.items())

        strategy_request = """
            Strategy Request:
            Based on the current state and feedback, provide a revised strategy for each player in Python dictionary format. 
            It's crucial to use the current positions of the players as the starting points for the new action plans. 
            The objective is to maximize the collective reward while considering the players' updated locations and past performance.
            Ensure it can be parsed by the `ast.literal_eval()` method and aligns with the mid-level goals and action functions.
            For example:

            {
                'player_0': {
                    'goal': 'Harvest Apple',
                    'action_plans': ['move_to((x1, y1), (x2, y2))', ...]
                },
                'player_1': {
                    'goal': 'Attack player_2',
                    'action_plans': ['move_to((x1, y1), (x2, y2))', 'fire_at((x2, y2))']
                },
                ...
            }
            The objective is to maximize the collective reward.
            """

        user_message = f"""Current State Description:
            - Map Size: {map_size} grid (Walls are located at the boundaries of the map).
            - Player Positions and Orientations: {player_positions}
            - Apple Locations: {apple_locations if 'apple' in state else []}
            - Beam Locations: {beam_locations if 'beam' in state else []}
            - Grass Locations: {grass_locations if 'grass' in state else []}

            Execution Outcomes:
            {execution_outcomes_str}

            Rewards:
            {rewards_str}

            {strategy_request}
            """
        return user_message

        
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