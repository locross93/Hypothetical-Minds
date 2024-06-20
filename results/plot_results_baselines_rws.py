# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:20:03 2024

@author: locro
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# load results/rws_scores.csv
df_plot = pd.read_csv('rws_scores.csv')

# reward per 1199 steps
df_plot['reward_per_1199'] = (df_plot['reward'] / df_plot['steps']) * 1199

# Initialize an empty DataFrame to store the results
filtered_df_plot = pd.DataFrame()

# Loop through each agent_type and scenario
# Define the agent types and scenarios
#agent_types = ['react', 'reflexion', 'hm']
agent_types = ['react', 'reflexion', 'planreact', 'hm']
scenarios = range(9)  # scenarios 0 to 8
for agent_type in agent_types:
    for scenario in scenarios:
        # Filter the DataFrame for the current agent_type and scenario
        filtered_df = df_plot[(df_plot['agent_type'] == agent_type) & (df_plot['scenario'] == scenario)]

        # Sort by datetime in descending order to get the latest entries first
        filtered_df = filtered_df.sort_values(by='datetime', ascending=False)
        
        count = len(filtered_df)
        print(f"Agent Type: {agent_type}, Scenario: {scenario}, Count: {count}")

        # If more than 5 entries, keep only the latest 5
        if count > 5:
            filtered_df = filtered_df.head(5)

        # Append to the result DataFrame
        filtered_df_plot = pd.concat([filtered_df_plot, filtered_df])

# Reset index of the final DataFrame
filtered_df_plot.reset_index(drop=True, inplace=True)

# add rl results
df_rl = pd.read_csv(f'running_with_scissors_in_the_matrix__repeated/eval_results_all_scenarios.csv')
df_rl['reward_per_1199'] = df_rl['reward']
df_rl['scenario_name'] = df_rl['scenario'].str.extract(r'(.+)_\d+$')[0]
df_rl['scenario'] = df_rl['scenario'].str.extract(r'(\d+)$')[0].astype(int)
filtered_df_plot = pd.concat([filtered_df_plot, df_rl])

# Define the mapping of old names to new names
name_mapping = {
    'ppo': 'PPO',
    'react': 'ReAct',
    'reflexion': 'Reflexion',
    'planreact': 'PlanReAct',
    'hm': 'Hypothetical\n Minds'
}

# Apply the mapping to the 'agent_type' column
filtered_df_plot['agent_type'] = filtered_df_plot['agent_type'].map(name_mapping)

sn_palette='Set2'
hue_order = ['PPO', 'ReAct', 'Reflexion', 'PlanReAct', 'Hypothetical\n Minds']
labels = ['0\nMixed\nStrategy', '1\nBest\nResponse', '2\n0 ∪ 1', '3\nFlip\nAfter 2', '4\nFlip\nAfter 1', '5\nGullible', '6\nRock', '7\nPaper', '8\nScissors']

save_file = 'rws_results_plot'

plt.figure(figsize=(13, 7))
sns.barplot(x='scenario', y='reward_per_1199', hue='agent_type', hue_order=hue_order, data=filtered_df_plot, palette=sn_palette, errorbar='se')
sns.despine(top=True, right=True)
plt.xticks(np.arange(len(labels)), labels, fontsize=18)
plt.title('Running With Scissors Repeated', fontsize=28, pad=25)
plt.ylabel('Reward', fontsize=28)
plt.xlabel('Evaluation Scenario', fontsize=28)
plt.tight_layout()
# Adjusting the legend position
plt.legend(loc='upper left', bbox_to_anchor=(-0.01, 1.055), fontsize=14, ncol=6)
plt.savefig(f'{save_file}.png', dpi=300)
plt.show()

scenario_grouped_df = filtered_df_plot.groupby(['agent_type', 'scenario']).agg(
    reward_per_1199_mean=('reward_per_1199', 'mean'),
    reward_per_1199_sem=('reward_per_1199', 'sem')
).reset_index()

grouped_df = filtered_df_plot.groupby(['agent_type']).agg(
    reward_per_1199_mean=('reward_per_1199', 'mean'),
    reward_per_1199_sem=('reward_per_1199', 'sem')
).reset_index()


plt.figure(figsize=(13, 7))
sns.barplot(x='agent_type', y='reward_per_1199', data=filtered_df_plot, palette=sn_palette, errorbar='se')
sns.despine(top=True, right=True)
plt.title('Prisoners Dilemma', fontsize=28, pad=25)
plt.ylabel('Reward', fontsize=28)
plt.xlabel('Model', fontsize=28)
plt.tight_layout()
# Adjusting the legend position
plt.legend(loc='upper left', bbox_to_anchor=(-0.01, 1.055), fontsize=14, ncol=6)
plt.show()
