# File: run_baseline.py
# File Created: Friday, 9th June 2023 3:26:23 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Tuesday, 11th July 2023 8:57:35 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Runs baseline Zero Shot test.

from chempe.data.lef_uspto import LEF_USPTO, DataVariants
from chempe.data.transforms import TransformStrToBrokenDownParts, TransformToRdKitIntermediates
from chempe.data.warnings import disable_warnings
from chempe.models.llm import generate_response_by_gpt, ModelVariants

import numpy as np
import openai
import os
import argparse
import tqdm

from pathlib import Path
from collections import defaultdict

def create_prompt(reactants, reagents):
    
    prompt = """You are an expert chemist. Your task is to predict the next intermediate molecules in the electron transfer process given the reactants, reagents and your experienced reaction prediction knowledge. There are some rules to follow.
1. Strictly follow the given format, and only respond with an atom mapped SMILES string.
2. Numbers immediately following : represent the atom mapping.
3. A . is used to distinguish between multiple molecules in the SMILES strings.
4. If no reagents exist, it is left blank.
5. The resulting intermediates must be be chemically reasonable and valid.
"""
    prompt += f"Reactants: {reactants}\n"
    prompt += f"Reagents: {reagents}\n"
    prompt += f"Intermediate:"
    return prompt

if __name__ == '__main__':
    
    # cmd line args
    parser = argparse.ArgumentParser(description="Runs a baseline test for few shot training")
    parser.add_argument("--gpt4", dest="use_gpt4", action="store_true", default = False, help = "Flag to use gpt4. Default: False (gpt3.5)")
    
    args = parser.parse_args()
    # disable rdkit warnings
    disable_warnings()
    
    # retrieve test data
    test_data = LEF_USPTO(DataVariants.TEST, transforms=[TransformStrToBrokenDownParts(), TransformToRdKitIntermediates()])
    
    # randomly select 20 test prompts
    n = int(os.environ['NSAMPLES'])
    test_indices = np.load(os.environ['TEST_IDX'], allow_pickle=True)[:n]
    
    # select model
    model = ModelVariants.GPT4 if args.use_gpt4 else ModelVariants.GPT3_5
    print(f"Using model {model}")
    
    # init openai
    openai.api_key = os.environ['OPEN_AI_KEY'] # loaded as environment variable
    
    # storage dir 
    results_dir = Path(os.environ['RESULTS_DIR']) / "step1_prediction_zs" / model.value
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # save information
    info = defaultdict(dict)
    # prompts, answers
    # {test: prompt}, {test:(ybar, y)}
    
    # run through openai api
    for test_idx in tqdm.tqdm(test_indices, desc = "Test Samples"):
        prompt = create_prompt(test_data[test_idx.item()].broken_down_parts.reactants, test_data[test_idx.item()].broken_down_parts.reagents)
        
        # get prompt
        predicted = generate_response_by_gpt(prompt=prompt,
                                             model_engine=ModelVariants.GPT4,
                                             temperature=float(os.environ['B_TEMP']),
                                             n=int(os.environ['B_PREDS']))
        
        # save results into dict
        info['prompts'][test_idx] = prompt
        info['ground_truth'][test_idx] = test_data[test_idx.item()].intermediates[0]
        info['predicted'][test_idx] = predicted
    
    # write to disk
    np.save(str(results_dir / "step1_prediction_zs_results.npy"), dict(info), allow_pickle=True)

