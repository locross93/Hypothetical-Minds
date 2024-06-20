import re
import abc
import ast
import heapq
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from llm_plan.agent import action_funcs


class Agent(abc.ABC):
    def __init__(
            self, 
            config: Dict[str, Any]) -> None:
        self.agent_id = config['agent_id']
        self.config = config

    def get_actions_from_plan(
            self, 
            goals_and_actions: Dict[str, Any],
            grid: np.ndarray) -> List[str]:
        """Given a plan, return a list of actions to be performed by the agent."""
        self.goal = goals_and_actions['goal']
        self.action_plans = goals_and_actions['action_plans']
        all_actions = []
        for action_plan in self.action_plans:
            split_idx = action_plan.find('(')
            func_name = action_plan[:split_idx]            
            func_args = ast.literal_eval(action_plan[split_idx+1:-1])
            func = getattr(action_funcs, func_name)            
            if func_name == 'move_to':
                start, goal = func_args
                # @TODO:
                #  if assert fails, then something is buggy with previously executed functions                
                # assert start == self.pos, \
                #     "Start position of move_to() function does not match agent's current position."
                paths, actions, current_orient = func(start, goal, grid, self.orientation)
                # update agent's position (if moved) and orientation
                if len(paths) > 0:
                    self.pos = paths[-1]
                self.orientation = current_orient
            elif func_name == 'fire_at':
                target = func_args
                actions, current_orient = func(self.pos, self.orientation, target)
                self.orientation = current_orient        
            all_actions.extend(actions)
        return all_actions

    def update_state(self, state: Dict[str, Any]) -> Optional[str]:
        """Update the position of the agent."""
        try:
            agent_key = [item for item in state['global'].keys() if item.startswith(self.agent_id)][0]        
            self.pos = state['global'][agent_key][0]    # (x, y) -> (col, row)
            self.orientation = agent_key.split('-')[1]
            if hasattr(self, 'goal'):                                
                # checking if agent is at it's goal location
                for action_plan in self.action_plans:
                    if 'move_to' in action_plan:
                        split_idx = action_plan.find('(')
                        func_args = ast.literal_eval(action_plan[split_idx+1:-1])
                        goal_pos = func_args[1]                                
                        output = f"Reached goal position {goal_pos}: {self.pos == goal_pos}"
                        return output
        except IndexError:
            print(f"Agent {self.agent_id} is dead for now...")
            if hasattr(self, 'goal'):                                
                # checking if agent is at it's goal location
                for action_plan in self.action_plans:
                    if 'move_to' in action_plan:
                        split_idx = action_plan.find('(')
                        func_args = ast.literal_eval(action_plan[split_idx+1:-1])
                        goal_pos = func_args[1]                                
                        output = f"Out of game now due to hit by laser in the last 5 steps."
                        return output
        