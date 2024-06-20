import subprocess

# Define the scenarios to loop through
#scenarios = [0, 1, 2, 3, 4, 5, 6, 7, 8]
scenarios = [2, 3, 4, 5, 6, 7, 8]

# Define the agent (either 'tom' or 'hierarchical')
agent = 'v2'

debug = False

# Loop through the scenarios and call the main script
for scenario in scenarios:
    print(f'Running scenario {scenario} with agent {agent}')
    if debug:
        subprocess.run(['python', '-m', 'pdb', 'main.py', '--agent', agent, '--scenario_num', str(scenario)])
    else:
        subprocess.run(['python', 'main.py', '--agent', agent, '--scenario_num', str(scenario)])
