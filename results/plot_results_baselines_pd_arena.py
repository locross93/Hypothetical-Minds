# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:34:31 2024

@author: locro
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# load results/rws_scores.csv
df_plot = pd.read_csv('pd_arena_scores.csv')

# reward per 1199 steps
df_plot['reward_per_1199'] = (df_plot['reward'] / df_plot['steps']) * 1199

# Initialize an empty DataFrame to store the results
filtered_df_plot = pd.DataFrame()

# Loop through each agent_type and scenario
# Define the agent types and scenarios
agent_types = ['react', 'reflexion', 'hm']
scenarios = [0, 3, 4]
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

# Define the mapping of old names to new names
name_mapping = {
    'react': 'React',
    'reflexion': 'Reflexion',
    'hm': 'HM'
}

# Apply the mapping to the 'agent_type' column
filtered_df_plot['agent_type'] = filtered_df_plot['agent_type'].map(name_mapping)

sn_palette='Set2'
labels = ['Cooperators', 'Grim\nReciprocators', 'Grim\nReciprocators\n(2 strikes)']

plt.figure(figsize=(13, 7))
sns.barplot(x='scenario', y='reward_per_1199', data=filtered_df_plot, hue='agent_type' , palette=sn_palette, errorbar='se')
sns.despine(top=True, right=True)
plt.xticks(np.arange(len(labels)), labels, fontsize=12, rotation=0)
plt.title('Prisoners Dilemma Arena - Hypothetical Minds', fontsize=28, pad=25)
plt.ylabel('Reward', fontsize=28)
plt.xlabel('Evaluation Scenario', fontsize=28)
plt.tight_layout()
# Adjusting the legend position
plt.legend(loc='upper left', bbox_to_anchor=(-0.01, 1.055), fontsize=14, ncol=6)
plt.show()
