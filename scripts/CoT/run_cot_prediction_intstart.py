# File: run_baseline.py
# File Created: Friday, 9th June 2023 3:26:23 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Thursday, 27th July 2023 8:10:12 am
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Runs baseline Few Shot test.

from chempe.data.warnings import disable_warnings
from chempe.models.llm import generate_response_by_gpt
from chempe.test_management.results import TestManager, Results, ResultVariants


import openai
import openai.error
import argparse
import tqdm
import os


def create_prompt(test, examples):
    
    prompt = """You are an expert chemist. Your task is to predict the resulting product in this heterolytic linear electron flow topology chemical reaction given the reactants, several examples including intermediate steps, and your experienced reaction prediction knowledge.Heterolytic linear electron flow topology involves pairs of electrons, and each intermediate electron transfer step alternates between remove bonds and add bonds. Additionally, there are some rules to follow.
1. Strictly follow the given format, and provide all intermediate molecules until the final product is reached.
2. Numbers immediately following : represent the atom mapping.
3. A . is used to distinguish between multiple molecules in the SMILES strings.
4. Each intermediate either adds or removes a bond from the previous step.
5. The resulting product must be be chemically reasonable and valid.
"""
    for idx, e in enumerate(examples):
        prompt += f"Reactants: {e.intermediates[1]}\n"
        for iidx, intermediate in enumerate(e.intermediates[2:]): # skip first, cause always the same
            prompt += f"Intermediate {iidx + 1}: {intermediate}\n"
        prompt += f"Products: {e.broken_down_parts.products}\n"
    prompt += f"Reactants: {test.intermediates[1]}\n"
    prompt += "Intermediate 1:"
    return prompt

if __name__ == '__main__':
    
    # cmd line args
    parser = argparse.ArgumentParser(description="Runs a baseline test for few shot training")
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
                      header="CoT",
                      model=TM.model,
                      scaffold=TM.scaffold,
                      filters=['intstart'],
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
                    print(f"error {e}")
                    train_data = train_data[:-1]
                else:
                    raise e
        # try parsing predicted
        rez = predicted[0].split("Products:")[-1].strip()
        
        

        results.store(test_indice=test_idx,
                      prompt=prompt,
                      gt=test_data.broken_down_parts.products,
                      predicted=[rez],
                      icl_indices=TM.icl_indices[idx])
        
    results.save()

