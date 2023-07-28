# File: run_baseline.py
# File Created: Friday, 9th June 2023 3:26:23 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Wednesday, 26th July 2023 12:35:28 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Attempts to use GPT to convert to kekulized form as step 1

from chempe.data.warnings import disable_warnings
from chempe.models.llm import generate_response_by_gpt
from chempe.test_management.results import TestManager, Results, ResultVariants


import openai
import argparse
import tqdm
import os


def create_prompt(test, examples):
    
    prompt = """You are an expert chemist. Your task is to kekulize the reactants given several examples and your experienced reaction prediction knowledge. Additionally, there are some rules to follow.
1. Strictly follow the given format, and only respond with an atom mapped SMILES string.
2. Numbers immediately following : represent the atom mapping.
3. A . is used to distinguish between multiple molecules in the SMILES strings.
5. The resulting kekulized form must be be chemically reasonable and valid.
"""
    for idx, e in enumerate(examples):
        prompt += f"Reactants: {e.broken_down_parts.reactants}\n"
        prompt += f"Kekulized: {e.intermediates[0]}\n"
    prompt += f"Reactants: {test.broken_down_parts.reactants}\n"
    prompt += "Kekulized:"
    return prompt

if __name__ == '__main__':
    
    # cmd line args
    parser = argparse.ArgumentParser(description="Attempts to use GPT to convert to kekulized form as step 1")
    parser.add_argument("-root", dest="root", default=".", help="Root Project Dir")
    parser.add_argument("--gpt35", dest="use_gpt35", action="store_true", default = False, help = "Flag to use gpt4. Default: False (gpt3.5)")
    parser.add_argument("--rand", dest='random', action="store_true", default=False, help = 'Use random ICL values. Default: False (Scaffold)')
    
    args = parser.parse_args()
    
    # disable rdkit warnings
    disable_warnings()
    
    TM = TestManager(proj_root=args.root,
                     gpt4=not args.use_gpt35,
                     scaffold=not args.random,
                     self_consistency=False)
    
    print(f"Using model {TM.model}")
    
    # openai key
    openai.api_key = os.environ['OPEN_AI_KEY']
    
    # storage dir 
    results = Results(proj_root=args.root,
                      header="step1",
                      model=TM.model,
                      scaffold=TM.scaffold,
                      filters=['base'],
                      variant=ResultVariants.VANILLA)
    
    # run through openai api
    for idx, (test_idx, test_data, train_data) in tqdm.tqdm(enumerate(TM.sample), desc="Test Samples", total = len(TM.sample)):
        while True:
            try:
                prompt = create_prompt(test_data, train_data)
                predicted = generate_response_by_gpt(prompt=prompt,
                                                    model_engine=TM.model,
                                                    temperature=TM.temp,
                                                    n=TM.num_preds)
                break
            except openai.error.InvalidRequestError as e:
                if "Please reduce your prompt; or completion length." in str(e):
                    # shorten by 1
                    print(e)
                    train_data = train_data[:-1]
                else:
                    raise e
            

        results.store(test_indice=test_idx,
                      prompt=prompt,
                      gt=test_data.intermediates[0],
                      predicted=predicted,
                      icl_indices=TM.icl_indices[idx])
        
    results.save()
        


