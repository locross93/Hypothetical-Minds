from main import run_main

# Define the scenarios to loop through
#scenarios = [0, 1, 2, 3, 4, 5, 6, 7, 8]
scenarios = [0, 2, 3, 5, 6, 7, 8]

# Define the agent (either 'tom' or 'hierarchical')
agent = 'v1'

debug = False

# Loop through the scenarios and call the main script
for scenario in scenarios:
    print(f'Running scenario {scenario} with agent {agent}')
    run_main(agent, scenario)
