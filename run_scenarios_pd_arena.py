import asyncio
import argparse
import sys
import numpy as np
import pandas as pd

async def run_main(agent_type, substrate, llm_type, num_seeds, scenarios):
    # Load the existing results DataFrame
    df_plot = pd.read_csv('results/pd_arena_scores.csv')

    # Loop through the scenarios and call the main script
    for scenario_num in scenarios:
        # Filter the DataFrame for the current agent_type and scenario
        filtered_df = df_plot[(df_plot['agent_type'] == agent_type) & (df_plot['scenario'] == scenario_num)]
        existing_seeds = len(filtered_df)

        # Calculate the number of additional seeds needed
        seeds_needed = num_seeds - existing_seeds

        # Generate additional seeds if needed
        if seeds_needed > 0:
            print(f"Generating {seeds_needed} additional seed(s) for Agent: {agent_type}, Scenario: {scenario_num}")
            for _ in range(seeds_needed):
                print(f"Agent: {agent_type}")
                print(f"Substrate: {substrate}")
                print(f"LLM Type: {llm_type}")
                print(f"Scenario: {scenario_num}")

                from main import main_async
                await main_async(substrate, scenario_num, agent_type, llm_type)
        else:
            print(f"No additional seeds needed for Agent: {agent_type}, Scenario: {scenario_num}")

async def main():
    parser = argparse.ArgumentParser(description="Run scenarios with specified parameters.")
    
    # Define default scenarios based on substrate
    default_scenarios = {
        "running_with_scissors_in_the_matrix__repeated": list(np.arange(9)),
        "running_with_scissors_in_the_matrix__arena": list(np.arange(8)),
        "prisoners_dilemma_in_the_matrix__repeated": list(np.arange(10)),
        "prisoners_dilemma_in_the_matrix__arena": [0 ,3, 4],
        "collaborative_cooking__asymmetric": list(np.arange(3)),
    }
    
    parser.add_argument("--agent", type=str, default="hm", help="Agent type (default: hm)")
    parser.add_argument("--substrate", type=str, default="pd_arena", help="Substrate type")
    parser.add_argument("--llm_type", type=str, default="gpt4", help="LLM type (default: gpt4)")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds (default: 1)")
    parser.add_argument("--scenarios", nargs='+', help="List of scenarios")
    
    args = parser.parse_args()

    substrate_dict = {
        'cc': 'collaborative_cooking__asymmetric',
        'rws': 'running_with_scissors_in_the_matrix__repeated',
        'pd': 'prisoners_dilemma_in_the_matrix__repeated',
        'rws_arena': 'running_with_scissors_in_the_matrix__arena',
        'pd_arena': 'prisoners_dilemma_in_the_matrix__arena'
    }
    substrate_name = substrate_dict[args.substrate]
    
    if args.scenarios is None:
        args.scenarios = default_scenarios.get(args.substrate, default_scenarios[substrate_name])
    
    await run_main(args.agent, substrate_name, args.llm_type, args.num_seeds, args.scenarios)

if __name__ == "__main__":
    asyncio.run(main())