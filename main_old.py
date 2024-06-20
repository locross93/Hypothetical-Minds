import argparse
import asyncio

import rws_tom_agent
import rws_eval

def run_main(agent, scenario_num):
    if agent == 'v1':
        asyncio.run(rws_eval.main_async(scenario_num))
    else:
        asyncio.run(rws_tom_agent.main_async(scenario_num, agent))

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run main_async from specified file with scenario number.')

    # Add the arguments
    parser.add_argument('--agent', 
                        type=str, 
                        choices=['v1', 'v2', 'v3', 'v4', 'hypothesis'],
                        help='The agent to run: "tom" for rws_tom_agent or "hierarchical" for rws_eval')

    parser.add_argument('--scenario_num', 
                        type=int, 
                        help='The scenario number to pass into main_async')

    parser.add_argument('--num_seeds', 
                        type=int, 
                        default=1,
                        help='The number of seeds to run for each scenario')

    # Execute the parse_args() method
    args = parser.parse_args()

    for i in range(args.num_seeds):
        print(f'Running scenario {args.scenario_num} with agent {args.agent} and seed {i}')
        run_main(args.agent, args.scenario_num)

if __name__ == "__main__":
    main()
