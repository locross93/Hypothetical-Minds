from typing import Any, Dict, List

import asyncio


 # The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    'gpt-4': 3e-05,
    'gpt-4-0613': 3e-05,
    'gpt-4-1106-preview': 1e-05,    # GPT4 Turbo
    'gpt-4-0125-preview': 1e-05,    # GPT4 Turbo
    'gpt-4o-2024-05-13': 5e-06,    # GPT4-o
    'gpt-3.5-turbo-1106': 1e-06,
    'meta-llama/Meta-Llama-3-70B-Instruct': 0.0,
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 0.0,

}
# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    'gpt-4': 6e-05,
    'gpt-4-0613': 6e-05,
    'gpt-4-1106-preview': 3e-05,    # GPT4 Turbo
    'gpt-4-0125-preview': 3e-05,    # GPT4 Turbo
    'gpt-4o-2024-05-13': 1.5e-05,    # GPT4-o
    'gpt-3.5-turbo-1106': 2e-06,
    'meta-llama/Meta-Llama-3-70B-Instruct': 0.0,
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 0.0,
}


class AsyncGPTController():
    """
    gpt-4 LLM wrapper for async API calls.
    llm: an instance of AsyncChatLLM,
    model_id: a unique id for the model to use
    model_args: arguments to pass to the api call
    """
    def __init__(
        self, 
        llm: Any,
        model_id: str,        
        **model_args,
    ) -> None:
        self.llm = llm
        self.model_id = model_id        
        self.model_args = model_args     
        self.all_responses = []
        self.total_inference_cost = 0

    def calc_cost(
        self, 
        response
    ) -> float:
        """
        Calculates the cost of a response from the openai API. Taken from https://github.com/princeton-nlp/SWE-bench/blob/main/inference/run_api.py

        Args:
        response (openai.ChatCompletion): The response from the API.

        Returns:
        float: The cost of the response.
        """
        model_name = response.model        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (
            MODEL_COST_PER_INPUT[model_name] * input_tokens
            + MODEL_COST_PER_OUTPUT[model_name] * output_tokens
        )
        return cost

    def get_prompt(
        self,
        system_message: str,
        user_message: str,
    ) -> List[Dict[str, str]]:
        """
        Get the (zero shot) prompt for the (chat) model.
        """    
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return messages
    
    async def get_response(
        self, 
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> Any:
        """
        Get the response from the model.
        """
        self.model_args['temperature'] = temperature
        self.model_args['model'] = self.llm.model
        return await self.llm(messages=messages, **self.model_args)
    
    async def run(
        self, 
        expertise: str,
        message: str,
        temperature: float,
    ) -> Dict[str, Any]:
        """Runs the Code Agent

        Args:
            expertise (str): The system message to use
            message (str): The user message to use

        Returns:
            A dictionary containing the code model's response and the cost of the performed API call
        """
        # Get the prompt
        messages = self.get_prompt(system_message=expertise, user_message=message)  
        # Get the response
        response = await self.get_response(messages=messages, temperature=temperature)
        # Get Cost
        cost = self.calc_cost(response=response)
        print(f"Cost for running {self.model_args['model']}: {cost}")
        # Store response including cost 
        if len(response.choices) == 1:
            response_str = response.choices[0].message.content
        else:
            # send list of responses when n > 1
            response_str = [choice.message.content for choice in response.choices]
        full_response = {
            'response': response,
            'response_str': response_str,
            'cost': cost
        }
        # Update total cost and store response
        self.total_inference_cost += cost
        self.all_responses.append(full_response)
    
        # Return response_string
        return full_response['response_str']
    
    async def batch_prompt_sync(
        self, 
        expertise: str, 
        messages: List[str],
        temperature: float,
    ) -> List[str]:
        """Handles async API calls for batch prompting.

        Args:
            expertise (str): The system message to use
            messages (List[str]): A list of user messages

        Returns:
            A list of responses from the code model for each message
        """
        responses = [self.run(expertise, message, temperature) for message in messages]
        return await asyncio.gather(*responses)

    def batch_prompt(
        self, 
        expertise: str, 
        messages: List[str], 
        temperature: float,
    ) -> List[str]:
        """=
        Synchronous wrapper for batch_prompt.

        Args:
            expertise (str): The system message to use
            messages (List[str]): A list of user messages
            temperature (str): The temperature to use for the API call

        Returns:
            A list of responses from the code model for each message
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(f"Loop is already running.")
        return loop.run_until_complete(self.batch_prompt_sync(expertise, messages, temperature))
    
    async def async_batch_prompt(self, expertise, messages, temperature=None):
        if temperature is None:
            temperature = self.model_args['temperature']        
        responses = [self.run(expertise, message, temperature) for message in messages]
        return await asyncio.gather(*responses)