import rws_gpt3_5
import asyncio

# Define the scenarios to loop through
#scenarios = [0, 1, 2, 3, 4, 5, 6, 7, 8]
scenarios = [7, 8]

# Define the agent (either 'tom' or 'hierarchical')
agent = 'v4'
gpt_version = '35'

num_seeds = 1

debug = False

# Loop through the scenarios and call the main script
for seed in range(num_seeds):
    for scenario in scenarios:
        print(f'Running scenario {scenario} with agent {agent}, seed {seed}')
        asyncio.run(rws_gpt3_5.main_async(scenario, agent, gpt_version))
