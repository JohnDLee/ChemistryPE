# File: llm.py
# File Created: Monday, 12th June 2023 3:49:12 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Monday, 12th June 2023 4:19:38 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Prompt retrieval from open ai

import openai
import time



# def generate_response_by_davinci(prompt, model_engine = 'text-davinci-003'):
#     time.sleep(2)
#     completion = openai.Completion.create(
#       engine=model_engine,
#       prompt=prompt,
#       temperature=0.5,
#       max_tokens=256,
# #       top_p=1.0,
#       frequency_penalty=0.0,
#       presence_penalty=0.0, 
#       n=5,
#     )
# #     message = completion.choices[0]['text'].strip()
    
#     message = completion.choices
#     message = [i['text'].strip() for i in message]
#     return message

def generate_response_by_gpt35(prompt, model_engine = "gpt-3.5-turbo", temperature = 0, n = 5):
    time.sleep(2)
    completion = openai.ChatCompletion.create(
        model=model_engine, 
        temperature=temperature, 
        n=n, 
        messages=[{"role": "user", "content": prompt}],
    )
#     message = completion.choices[0].message.content.strip()

    message = completion.choices
    message = [i.message.content.strip() for i in message]
    return message

# def generate_response_by_gpt4(prompt):
#     # Create a Steamship client
#     # NOTE: When developing a package, just use `self.client`
#     time.sleep(2)
#     client = Steamship(workspace="gpt-4-92g")

#     # Create an instance of this generator
#     generator = client.use_plugin('gpt-4', config={"temperature":0.5, "n": 5})

#     # Generate text
#     task = generator.generate(text=prompt)
#     # Wait for completion of the task.
#     task.wait()
#     # Print the output
# #     message = task.output.blocks[0].text.strip()
#     message = task.output.blocks
#     message = [i.text.strip() for i in message]
#     return message