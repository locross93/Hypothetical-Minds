# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:19:10 2024

@author: locro
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define the substrate names and their corresponding CSV files
substrates = ['rws', 'rws_arena', 'cc', 'pd']
csv_files = ['rws_scores.csv', 'rws_arena_scores.csv', 'cc_scores.csv', 'pd_scores.csv']

substrate_dict = {
    'cc': 'Collaborative Cooking Asymmetric',
    'rws': 'Running with Scissors Repeated',
    'pd': 'Prisoners Dilemma Repeated',
    'rws_arena': 'Running with Scissors Arena',
    'pd_arena': 'Prisoners Dilemma Arena'
}

substrate_dict2 = {
    'cc': 'collaborative_cooking__asymmetric',
    'rws': 'running_with_scissors_in_the_matrix__repeated',
    'pd': 'prisoners_dilemma_in_the_matrix__repeated',
    'rws_arena': 'running_with_scissors_in_the_matrix__arena',
    'pd_arena': 'prisoners_dilemma_in_the_matrix__arena'
}

# Define default scenarios based on substrate
default_scenarios = {
    "Running with Scissors Repeated": list(np.arange(9)),
    "Running with Scissors Arena": list(np.arange(8)),
    "Prisoners Dilemma Repeated": list(np.arange(10)),
    "Prisoners Dilemma Arena": [0 ,3, 4],
    "Collaborative Cooking Asymmetric": list(np.arange(3)),
}

# Initialize an empty DataFrame to store the results
filtered_df_plot = pd.DataFrame()

# Loop through each substrate and load the corresponding CSV file
agent_types = ['react', 'reflexion', 'planreact', 'hm']
for substrate, csv_file in zip(substrates, csv_files):
    df_plot = pd.read_csv(csv_file)
    
    # Reward per 1199 steps
    df_plot['reward_per_1199'] = (df_plot['reward'] / df_plot['steps']) * 1199
    
    # Add a column for the substrate name
    substrate_name = substrate_dict[substrate]
    df_plot['substrate'] = substrate_name
    
    scenarios = default_scenarios[substrate_name]
    for agent_type in agent_types:
        for scenario in scenarios:
            # Filter the DataFrame for the current agent_type and scenario
            filtered_df = df_plot[(df_plot['agent_type'] == agent_type) & (df_plot['scenario'] == scenario)]
            
            # Sort by datetime in descending order to get the latest entries first
            filtered_df = filtered_df.sort_values(by='datetime', ascending=False)
            
            # If more than 5 entries, keep only the latest 5
            count = len(filtered_df)
            if count > 5:
                filtered_df = filtered_df.head(5)
        
            # Append to the result DataFrame
            filtered_df_plot = pd.concat([filtered_df_plot, filtered_df])
            
    # add rl results to df
    full_substrate = substrate_dict2[substrate]
    df_rl = pd.read_csv(f'{full_substrate}/eval_results_all_scenarios.csv')
    df_rl['substrate'] = substrate_name
    df_rl['reward_per_1199'] = df_rl['reward']
    filtered_df_plot = pd.concat([filtered_df_plot, df_rl])

# Define the mapping of old names to new names
name_mapping = {
    'ppo': 'PPO',
    'react': 'ReAct',
    'reflexion': 'Reflexion',
    'planreact': 'PlanReAct',
    'hm': 'HM'
}

# Apply the mapping to the 'agent_type' column
filtered_df_plot['agent_type'] = filtered_df_plot['agent_type'].map(name_mapping)
hue_order = ['PPO', 'ReAct', 'Reflexion', 'PlanReAct', 'HM']
width = 0.8  # Decrease the width of the bars

# Create subplots for each substrate
save_file = 'results_by_model_all_envs'
fig, axes = plt.subplots(1, len(substrates), figsize=(40, 8), sharey=False)  # Increase the figure width
sn_palette = 'Set2'

for i, substrate in enumerate(substrates):
    substrate_name = substrate_dict[substrate]
    
    # Filter data for the current substrate
    substrate_df = filtered_df_plot[filtered_df_plot['substrate'] == substrate_name]
    
    # Create the barplot for the current substrate
    sns.barplot(x='agent_type', y='reward_per_1199', data=substrate_df, palette=sn_palette, order=hue_order, ax=axes[i], width=width, errorbar='se')
    sns.despine(top=True, right=True)
    axes[i].set_title(substrate_name, fontsize=24)
    axes[i].set_xlabel('Model Type', fontsize=20)
    axes[i].set_ylabel('Average Reward', fontsize=30)
    axes[i].tick_params(labelsize=20)

plt.tight_layout(pad=5.0)
if save_file is not None:
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
plt.show()

grouped_df = filtered_df_plot.groupby(['agent_type', 'substrate', 'scenario']).agg(
    reward_per_1199_mean=('reward_per_1199', 'mean'),
    reward_per_1199_sem=('reward_per_1199', 'sem')
).reset_index()

# Save the resulting DataFrame to a CSV file
grouped_df.to_csv('results_by_model_all_envs.csv', index=False)