import os
import time
import json
import asyncio
import argparse
import datetime
import importlib
import subprocess

from llm_plan.agent.agent_config import agent_config
from llm_plan.env.mp_llm_env import MeltingPotLLMEnv
from llm_plan.controller.async_llm import AsyncChatLLM
from llm_plan.controller.async_gpt_controller import AsyncGPTController

def setup_environment(substrate_name, scenario_num):
    sprite_label_path = f'./llm_plan/sprite_labels/{substrate_name}'
    env = MeltingPotLLMEnv(substrate_name, sprite_label_path, scenario_num)
    return env

def setup_agent(api_key, model_id, model_settings, substrate, agent_type, llm_type='gpt4'):
    if llm_type == 'gpt4':
        llm = AsyncChatLLM(kwargs={'api_key': api_key, 'model': 'gpt-4-1106-preview'})
        controller = AsyncGPTController(
            llm=llm,
            model_id=model_id,
            **model_settings
        )
    elif llm_type == 'gpt35':
        llm = AsyncChatLLM(kwargs={'api_key': api_key, 'model': 'gpt-3.5-turbo-1106'})
        controller = AsyncGPTController(
            llm=llm,
            model_id=model_id,
            **model_settings
        )
    elif llm_type == 'llama3':
        kwargs = {
            'api_key': "EMPTY",
            'base_url': "http://localhost",
            'port': 8000,
            'version': 'v1',
            'model': 'meta-llama/Meta-Llama-3-70B-Instruct'
        }
        llm = AsyncChatLLM(kwargs=kwargs)
        controller = AsyncGPTController(
            llm=llm,
            model_id=model_id,
            **model_settings
        )
    elif llm_type == 'mixtral':
        kwargs = {
            'api_key': "EMPTY",
            'base_url': "http://localhost",
            'port': 8000,
            'version': 'v1',
            'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        }
        llm = AsyncChatLLM(kwargs=kwargs)
        controller = AsyncGPTController(
            llm=llm,
            model_id=model_id,
            **model_settings
        )


    agent_config_obj = {'agent_id': model_id}
    
    agent_class_path = agent_config[substrate][agent_type][llm_type]
    agent_module_path, agent_class_name = agent_class_path.rsplit('.', 1)
    agent_module = importlib.import_module(agent_module_path)
    agent_class = getattr(agent_module, agent_class_name)
    
    if 'hypothetical_minds' in agent_type or 'hm' in agent_type:
        agent_config_obj['self_improve'] = True
    
    agent = agent_class(agent_config_obj, controller)
    return agent

async def main_async(substrate_name, scenario_num, agent_type, llm_type):
    if llm_type == 'gpt4' or llm_type == 'gpt35':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("No API key found. Please set the OPENAI_API_KEY environment variable.")
    else:
        api_key = "EMPTY"
    if llm_type == 'gpt4':
        model_settings = {
            "model": "gpt-4-1106-preview",
            "max_tokens": 4000,
            "temperature": 0.2,
            "top_p": 1.0,
            "n": 1,
        }
    elif llm_type == 'gpt35':
        model_settings = {
            "model": "gpt-3.5-turbo-1106",
            "max_tokens": 2000,
            "temperature": 0.2,
            "top_p": 1.0,
            "n": 1,
        }
    elif llm_type == 'llama3':
        model_settings = {
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "max_tokens": 2000,
            "temperature": 0.2,
            "top_p": 1.0,
            "n": 10,
        }
    elif llm_type == 'mixtral':
        model_settings = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 1.0,
            "n": 10,
        }

    agent = setup_agent(api_key, model_id=f"player_0", model_settings=model_settings, substrate=substrate_name, agent_type=agent_type, llm_type=llm_type)
    agent.agent_type = agent_type 
    agent.llm_type = llm_type
    
    env = setup_environment(substrate_name, scenario_num)

    run_episode_module = importlib.import_module(f"environments.{substrate_name}")
    run_episode = run_episode_module.run_episode
    
    frame_folder = await run_episode(env, agent)

    # Save video of the frames
    create_video_script = './create_videos.sh'
    subprocess.call([create_video_script, frame_folder])

def main():
    parser = argparse.ArgumentParser(description='Run the multi-agent environment.')
    parser.add_argument('--substrate', type=str, required=True, help='Substrate name')
    parser.add_argument('--scenario_num', type=int, required=True, help='Scenario number')
    parser.add_argument('--agent_type', type=str, default='hm', help='Agent type')
    parser.add_argument('--llm_type', type=str, default='gpt4', help='LLM Type')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of seeds')
    args = parser.parse_args()

    substrate_dict = {
        'cc': 'collaborative_cooking__asymmetric',
        'rws': 'running_with_scissors_in_the_matrix__repeated',
        'pd': 'prisoners_dilemma_in_the_matrix__repeated',
        'rws_arena': 'running_with_scissors_in_the_matrix__arena',
    }
    substrate_name = substrate_dict[args.substrate]
    
    loop = asyncio.get_event_loop()
    for seed in range(args.num_seeds):
        loop.run_until_complete(main_async(substrate_name, args.scenario_num, args.agent_type, args.llm_type))

if __name__ == "__main__":
    main()