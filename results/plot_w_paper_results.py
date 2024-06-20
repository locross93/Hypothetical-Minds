# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 19:04:01 2024

@author: locro
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# @title Load scenario results
path = 'https://storage.googleapis.com/dm-meltingpot/meltingpot-results-2.1.1.feather'  # @param {type: 'string'}

def load_scenario_results(path):
  results = pd.read_feather(path)
  # Drop training scores
  scenario_results = results.drop(
      labels=set(results.substrate.unique()),
      axis=1,
      errors='ignore')
  return scenario_results

# Function to get the mapla with the highest focal_per_capita_return for each scenario
def get_max_focal_mapla(group):
    return group.loc[group['focal_per_capita_return'].idxmax()]


scenario_results = load_scenario_results(path)

# Filter rows where 'scenario' column contains 'running_with_scissors__repeated'
rws_results = scenario_results[scenario_results['scenario'].str.contains('running_with_scissors_in_the_matrix__repeated')]

# Extract the scenario number and create a new column 'scenario_number'
rws_results['scenario'] = rws_results['scenario'].str.extract('running_with_scissors_in_the_matrix__repeated_(\d+)')[0]

# Convert scenario_number column to numeric type (optional)
rws_results['scenario'] = pd.to_numeric(rws_results['scenario'])

# Splitting the DataFrame into exploiters and non-exploiters
exploiters = rws_results[rws_results['mapla'].str.contains('exploiter')]
non_exploiters = rws_results[~rws_results['mapla'].str.contains('exploiter')]

# Grouping by 'scenario' and applying the function
max_exploiters = exploiters.groupby('scenario').apply(get_max_focal_mapla)
max_non_exploiters = non_exploiters.groupby('scenario').apply(get_max_focal_mapla)

# Renaming columns for max_exploiters
max_exploiters = max_exploiters.rename(columns={'focal_per_capita_return': 'reward', 'scenario_num': 'scenario'})
max_exploiters['agent_type'] = 'RL Exploiter*'

# Renaming columns for max_non_exploiters
max_non_exploiters = max_non_exploiters.rename(columns={'focal_per_capita_return': 'reward', 'scenario_num': 'scenario'})
max_non_exploiters['agent_type'] = 'RL Baseline'

# load results/rws_scores.csv
df = pd.read_csv('rws_scores.csv')

# remove rows where reward = 0
df_v4 = df[df['agent_type'] == 'v4']
df_v4['reward'] = (df_v4['reward'] / df_v4['steps']) * 1199
df_v4['agent_type'] = 'LLM Agent'
df_v4['datetime'] = pd.to_datetime(df_v4['datetime'], format='%Y-%m-%d_%H-%M-%S')
# Define the cutoff datetime
cutoff_datetime = pd.Timestamp('2024-01-26 20:00:00')
# Create a mask for rows to keep
mask = ~((df_v4['datetime'] < cutoff_datetime))
# Apply the mask to the DataFrame
df_v4 = df_v4[mask]
# Keep only the rows where 'gpt_version' is NaN - the default
df_v4 = df_v4[df_v4['gpt_version'].isna()]

# Concatenate the two DataFrames
df_plot = pd.concat([df_v4, max_exploiters, max_non_exploiters])

save_file = 'reward_evals_vs_rl'

# Plotting
sn_palette='Set2'
hue_order = ['LLM Agent', 'RL Baseline', 'RL Exploiter*']

plt.figure(figsize=(12, 6))
sns.barplot(x='scenario', y='reward', hue='agent_type', data=df_plot, palette=sn_palette, hue_order=hue_order)
sns.despine(top=True, right=True)
plt.xticks(fontsize=18)
plt.title('Reward per Episode', fontsize=24)
plt.ylabel('Reward', fontsize=24)
plt.xlabel('Evaluation Scenario', fontsize=24)
plt.tight_layout()
# Adjusting the legend position
plt.legend(loc='upper left', bbox_to_anchor=(0, 1.05), fontsize=14)
if save_file is not None:
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
plt.show()
