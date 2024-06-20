import re
import pandas as pd
import os
import datetime

def parse_output_with_good_hypothesis(file_path):
    # Extract scenario number and datetime from the file path
    parts = file_path.split('/')
    scenario_part = parts[-2]  # Assuming the scenario folder is the second last part of the file path
    scenario_number = scenario_part.split('_')[1]
    datetime_str = '_'.join(scenario_part.split('_')[-2:])

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    interactions = []
    ghf_nums = []
    good_hypothesis_found_at = None
    hypotheses = {}   # Track first occurrences of hypotheses
    last_ghf_num = None  # To track the last interaction where a good hypothesis was found
    hypothesis_text = None

    for i, line in enumerate(lines):
        if 'Interaction' in line:
            interaction_info = re.findall(r'Interaction (\d+), .*?rewards=(-?\d+\.\d+)', line)
            if interaction_info:
                interaction_number, reward = interaction_info[0]
                interactions.append({
                    'interaction': int(interaction_number),
                    'reward': float(reward)
                })
                
        elif 'Good hypothesis found' in line:
            for j in range(i, -1, -1):
                if 'Interaction' in lines[j]:
                    interaction_info = re.findall(r'Interaction (\d+),', lines[j])
                    if interaction_info:
                        ghf_num = int(interaction_info[0])
                        if not good_hypothesis_found_at:
                            good_hypothesis_found_at = ghf_num
                        # Append to ghf_nums if it's a continuous streak
                        if last_ghf_num is None or ghf_num == last_ghf_num + 1:
                            ghf_nums.append(ghf_num)
                        last_ghf_num = ghf_num
                        break

            hypothesis_info = re.search(r"'Opponent_strategy': '(.+?)',", line)
            if hypothesis_info:
                hypothesis_text = hypothesis_info.group(1)
                # If this hypothesis hasn't been recorded yet, find its first occurrence
                if hypothesis_text not in hypotheses:
                    # Search backward for the interaction number where the hypothesis was first mentioned
                    for j in range(0, len(lines)):
                        if hypothesis_text in lines[j]:
                            # Continue searching backward for the interaction number
                            for k in range(j, -1, -1):
                                interaction_match = re.search(r'Interaction (\d+),', lines[k])
                                if interaction_match:
                                    hypotheses[hypothesis_text] = int(interaction_match.group(1))
                                    break
                            break

    # Convert interactions to DataFrame
    df_interactions = pd.DataFrame(interactions)
    if good_hypothesis_found_at is not None:
        df_interactions['relative_interaction'] = df_interactions.apply(lambda row: row['interaction'] - good_hypothesis_found_at if row['interaction'] <= good_hypothesis_found_at or row['interaction'] in ghf_nums else None, axis=1)
    if hypothesis_text is not None:
        df_interactions['hyp_discovered_before'] = hypotheses[hypothesis_text] - good_hypothesis_found_at
    
    # Add scenario number and datetime as columns
    df_interactions['scenario_number'] = scenario_number
    df_interactions['datetime'] = datetime_str

    return df_interactions

# Diagnostic code to check the directory and file processing
cutoff_datetime_str = '2024-01-26_20-00-00'
cutoff_datetime = datetime.datetime.strptime(cutoff_datetime_str, '%Y-%m-%d_%H-%M-%S')
root_dir = '/ccn2/u/locross/llmv_marl/frames/running_with_scissors_in_the_matrix__repeated/agent_v4/'
all_interactions_df = pd.DataFrame()

for dir in os.listdir(root_dir):
    if dir.startswith('scenario_'):
        dir_datetime_str = dir.split('_')[-2:]
        dir_datetime_str = '_'.join(dir_datetime_str)  # Join the date and time parts
        try:
            dir_datetime = datetime.datetime.strptime(dir_datetime_str, '%Y-%m-%d_%H-%M-%S')
            if dir_datetime >= cutoff_datetime:
                file_path = os.path.join(root_dir, dir, 'output_data.txt')
                if os.path.exists(file_path):
                    df_interactions = parse_output_with_good_hypothesis(file_path)
                    all_interactions_df = pd.concat([all_interactions_df, df_interactions])
        except ValueError as e:
            print(f"Error parsing date from directory name {dir}: {e}")

all_interactions_df.reset_index(drop=True, inplace=True)
all_interactions_df.head()

# save dataframe
all_interactions_df.to_csv('df_reward_by_good_hypothesis2.csv')