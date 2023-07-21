# File: llm.py
# File Created: Monday, 12th June 2023 3:49:12 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Tuesday, 11th July 2023 9:18:16 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Prompt retrieval from open ai

import openai
import time
import enum

class ModelVariants(enum.Enum):
    GPT3_5='gpt-3.5-turbo'
    GPT4='gpt-4'

def generate_response_by_gpt(prompt, model_engine: ModelVariants, temperature = 0.5, n = 5, retries = 6):
    """Attempts to generate a resmpont by gpt, retrying using exponential wait

    Args:
        prompt (str): prompt to send
        model_engine (ModelVariants): model enginer to use
        temperature (float, optional): temperature to use of model from 0-2. Larger values means more randomness. Defaults to 0.5.
        n (int, optional): Number of outputs to provide. Defaults to 6 --> 2^6 = 1min.
    """
    error = None
    sleep_time = 1
    for i in range(retries):
        try:
            completion = openai.ChatCompletion.create(
                model=model_engine.value, 
                temperature=temperature, 
                n=n, 
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:
            print(f"An exception occurred ({e}). Retrying after {sleep_time} seconds.")
            time.sleep(sleep_time)
            sleep_time *= 2
            error = e
    else:
        # failed
        raise RuntimeError(error)

    message = completion.choices
    message = [i.message.content.strip() for i in message]
    
    # returns all n output choices as a list
    return message