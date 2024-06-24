# Hypothetical-Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models

## Overview
Hypothetical Minds is an autonomous LLM-based agent for diverse multi-agent settings, integrating a Theory of Mind module that scaffolds the high-level planning process by generating, evaluating, and refining hypotheses about other agents’ strategies in natural language.

## Running Hypothetical Minds and Baselines

To run an episode of Hypothetical Minds, use main.py as in the following example with "Running With Scissors Repeated":

```bash
python main.py --substrate rws --scenario_num 0 --agent_type hm --llm_type gpt4
```

To loop through every scenario in a substrate, use run_scenarios.py as in the following example running the Reflexion baseline on Collaborative Cooking Asymmetric:

```bash
python run_scenarios.py --agent planreact --substrate cc --num_seeds 5
```