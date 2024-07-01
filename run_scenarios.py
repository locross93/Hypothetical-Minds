import asyncio
import argparse
import sys
import numpy as np

async def run_main(agent_type, substrate, llm_type, num_seeds, scenarios):
    # Loop through the scenarios and call the main script
    for s in range(num_seeds):
        for scenario_num in scenarios:
            print(f"Agent: {agent_type}")
            print(f"Substrate: {substrate}")
            print(f"LLM Type: {llm_type}")
            print(f"Scenario: {scenario_num}")

            from main import main_async
            await main_async(substrate, scenario_num, agent_type, llm_type)

async def main():
    parser = argparse.ArgumentParser(description="Run scenarios with specified parameters.")
    
    # Define default scenarios based on substrate
    default_scenarios = {
        "running_with_scissors_in_the_matrix__repeated": list(np.arange(9)),
        "running_with_scissors_in_the_matrix__arena": list(np.arange(8)),
        "prisoners_dilemma_in_the_matrix__repeated": list(np.arange(10)),
        "collaborative_cooking__asymmetric": list(np.arange(3)),
    }
    
    parser.add_argument("--agent", type=str, default="hm", help="Agent type (default: hm)")
    parser.add_argument("--substrate", type=str, required=True, help="Substrate type")
    parser.add_argument("--llm_type", type=str, default="gpt4", help="LLM type (default: gpt4)")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of seeds (default: 1)")
    parser.add_argument("--scenarios", nargs='+', help="List of scenarios")
    
    args = parser.parse_args()

    substrate_dict = {
        'cc': 'collaborative_cooking__asymmetric',
        'rws': 'running_with_scissors_in_the_matrix__repeated',
        'pd': 'prisoners_dilemma_in_the_matrix__repeated',
        'rws_arena': 'running_with_scissors_in_the_matrix__arena',
    }
    substrate_name = substrate_dict[args.substrate]
    
    if args.scenarios is None:
        args.scenarios = default_scenarios.get(args.substrate, default_scenarios[substrate_name])
    
    await run_main(args.agent, substrate_name, args.llm_type, args.num_seeds, args.scenarios)

if __name__ == "__main__":
    asyncio.run(main())