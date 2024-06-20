import asyncio
import argparse

async def run_main(agent, scenario, substrate, llm_type):
    parser = argparse.ArgumentParser(description='Run the multi-agent environment.')
    parser.add_argument('--substrate', type=str, default=substrate, help='Substrate name')
    parser.add_argument('--scenario_num', type=int, default=scenario, help='Scenario number')
    parser.add_argument('--agent_type', type=str, default=agent, help='Agent type')
    parser.add_argument('--llm_type', type=str, default=llm_type, help='LLM Type')
    args = parser.parse_args()

    from main import main_async
    await main_async(args.substrate, args.scenario_num, args.agent_type, args.llm_type)

# Define the scenarios to loop through
scenarios = [0, 3, 4]

# Define the agent (either 'tom' or 'hierarchical')
agent = 'hm'
#agent = 'reflexion'
#agent = 'react'

# Define the substrate and LLM type
substrate = 'prisoners_dilemma_in_the_matrix__arena'
llm_type = 'gpt4'

async def main():
    # Loop through the scenarios and call the main script
    for scenario in scenarios:
        print(f'Running scenario {scenario} with agent {agent}')
        await run_main(agent, scenario, substrate, llm_type)

if __name__ == "__main__":
    asyncio.run(main())
