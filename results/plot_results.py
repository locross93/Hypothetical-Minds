import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

save_file = 'ablations_reward_evals'

# load results/rws_scores.csv
df = pd.read_csv('rws_scores.csv')

# remove rows where reward = 0
df_plot = df[df['reward'] != 0]

# reward per 1199 steps
df_plot['reward_per_1199'] = (df_plot['reward'] / df_plot['steps']) * 1199

df_plot['datetime'] = pd.to_datetime(df_plot['datetime'], format='%Y-%m-%d_%H-%M-%S')

# Define the cutoff datetime
cutoff_datetime = pd.Timestamp('2024-01-26 20:00:00')

# Create a mask for rows to keep
mask = ~((df_plot['agent_type'] == 'v4') & (df_plot['datetime'] < cutoff_datetime))

# Apply the mask to the DataFrame
df_plot = df_plot[mask]

# Keep only the rows where 'gpt_version' is NaN - the default
df_plot = df_plot[df_plot['gpt_version'].isna()]

# Initialize an empty DataFrame to store the results
filtered_df_plot = pd.DataFrame()

# Loop through each agent_type and scenario
# Define the agent types and scenarios
agent_types = ['v1', 'v2', 'v3', 'v4']
scenarios = range(9)  # scenarios 0 to 8
for agent_type in agent_types:
    for scenario in scenarios:
        # Filter the DataFrame for the current agent_type and scenario
        filtered_df = df_plot[(df_plot['agent_type'] == agent_type) & (df_plot['scenario'] == scenario)]

        # Sort by datetime in descending order to get the latest entries first
        filtered_df = filtered_df.sort_values(by='datetime', ascending=False)

        # If more than 5 entries, keep only the latest 5
        if len(filtered_df) > 5:
            filtered_df = filtered_df.head(5)

        # Append to the result DataFrame
        filtered_df_plot = pd.concat([filtered_df_plot, filtered_df])

# Reset index of the final DataFrame
filtered_df_plot.reset_index(drop=True, inplace=True)

# Plotting
sn_palette='Set2'
hue_order = ['v1', 'v2', 'v3', 'v4']

plt.figure(figsize=(12, 6))
sns.barplot(x='scenario', y='reward_per_1199', hue='agent_type', data=filtered_df_plot, palette=sn_palette, hue_order=hue_order, errorbar='se')
sns.despine(top=True, right=True)
plt.xticks(fontsize=18)
plt.title('Reward per Episode', fontsize=24)
plt.ylabel('Reward', fontsize=24)
plt.xlabel('Evaluation Scenario', fontsize=24)
plt.tight_layout()
# Adjusting the legend position
plt.legend(loc='upper left', bbox_to_anchor=(0, 1.08), fontsize=14, ncol=2)
if save_file is not None:
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
plt.show()


# ### HOW MANY SEEDS DO WE HAVE PER AGENT, SCENARIO
# # Define the agent types and scenarios
# agent_types = ['v1', 'v2', 'v3', 'v4']
# scenarios = range(9)  # scenarios 0 to 8

# # Loop through each agent_type and scenario
# for agent_type in agent_types:
#     for scenario in scenarios:
#         # Filter the DataFrame for the current agent_type and scenario
#         filtered_df = df_plot[(df_plot['agent_type'] == agent_type) & (df_plot['scenario'] == scenario)]

#         # Get the count of elements (rows) in the filtered DataFrame
#         count = len(filtered_df)

#         # Print the count
#         print(f"Agent Type: {agent_type}, Scenario: {scenario}, Count: {count}")