# File: run_baseline.py
# File Created: Friday, 9th June 2023 3:26:23 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Friday, 21st July 2023 2:00:56 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Runs baseline Few Shot test.

from chempe.data.lef_uspto import LEF_USPTO, DataVariants
from chempe.data.transforms import TransformStrToBrokenDownParts, TransformToRdKitIntermediates
from chempe.data.warnings import disable_warnings
from chempe.models.llm import generate_response_by_gpt, ModelVariants

import numpy as np
import openai
import argparse
import tqdm
import os

from pathlib import Path
from collections import defaultdict

def create_prompt(test, examples):
    
    prompt = """You are an expert chemist. Your task is to predict the next intermediate product in the electron transfer process or the final product in a reaction given the reactants, reagents, previous intermediates, several examples including intermediate steps, and your experienced reaction prediction knowledge. There are some rules to follow.
1. Strictly follow the given format, and only respond with an atom mapped SMILES string of the next intermediate.
2. Numbers immediately following : represent the atom mapping.
3. A . is used to distinguish between multiple molecules in the SMILES strings.
4. If no reagents exist, it is left blank.
5. The resulting intermediate must be be chemically reasonable and valid.
6. If result is the final product of the chemical reaction, append a ? to the start of the SMILES string.
"""
    for idx, e in enumerate(examples):
        prompt += f"Reactants: {e.broken_down_parts.reactants}\n"
        prompt += f"Reagents: {e.broken_down_parts.reagents}\n"
        for iidx, intermediate in enumerate(e.intermediates[1:]): # skip first, cause always the same
            prompt += f"Intermediate {iidx + 1}: {intermediate}\n"
        prompt += f"Products: {e.broken_down_parts.products}\n"
    prompt += f"Reactants: {test.broken_down_parts.reactants}\n"
    prompt += f"Reagents: {test.broken_down_parts.reagents}\n"
    prompt += "Intermediate 1:"
    return prompt

def add_prompt(prompt, intermediates: list):
    prompt += f' {intermediates[-1]}\nIntermediate {len(intermediates) + 1}:'
    return prompt

if __name__ == '__main__':
    
    # cmd line args
    parser = argparse.ArgumentParser(description="Runs a single step prediction test")
    parser.add_argument("--gpt4", dest="use_gpt4", action="store_true", default = False, help = "Flag to use gpt4. Default: False (gpt3.5)")
    
    args = parser.parse_args()
    
    # disable rdkit warnings
    disable_warnings()
    
    # retrieve train/test data
    train_data = LEF_USPTO(DataVariants.TRAIN, transforms=[TransformStrToBrokenDownParts(), TransformToRdKitIntermediates()])
    test_data = LEF_USPTO(DataVariants.TEST, transforms=[TransformStrToBrokenDownParts(), TransformToRdKitIntermediates()])
    

    # select n test prompts
    n = 2#int(os.environ['NSAMPLES'])
    test_indices = np.load(os.environ['TEST_IDX'], allow_pickle=True)[:n]

    # set ICL samples
    k = int(os.environ['KICL'])
    icl_indices = np.load(os.environ['ICL_IDX'], allow_pickle=True)
    
    # select model
    model = ModelVariants.GPT4 if args.use_gpt4 else ModelVariants.GPT3_5
    print(f"Using model {model}")
    
    # init openai
    openai.api_key = os.environ['OPEN_AI_KEY'] # loaded as environment variable
    
    # storage dir 
    results_dir = Path(os.environ['RESULTS_DIR']) / "cot_prediction_step" / model.value
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # save information
    info = defaultdict(dict)
    # indices, prompts, answers
    # {test: [train]}, {test: prompt}, {test:(ybar, y)}
    
    # run through openai api
    p = True
    for test_idx in tqdm.tqdm(test_indices, desc="Test Samples", ):
        train_indices = icl_indices[test_idx][:k]
        prompt = create_prompt(test_data[test_idx.item()], train_data[train_indices])
        intermediates = []
        while True:
            predicted = generate_response_by_gpt(prompt=prompt,
                                                model_engine=model,
                                                temperature=float(os.environ['B_TEMP']),
                                                n=int(os.environ['B_PREDS']))
            if predicted[0][0] == "?": # if final product
                print("Final product found.")
                predicted[0] = predicted[0][1:] # remove ?
                break
            intermediates.append(predicted[0])
            print(f"Predicted {len(intermediates)} intermediates.")
            if len(intermediates) > 10:
                print("10 intermediates were found without a final product, giving up.")
                break
            
            prompt = add_prompt(prompt, intermediates)
            
        if p:
            print(prompt)
            p = False
            

        info['icl_indices'][test_idx] = train_indices
        info['intermediates'][test_idx] = intermediates
        info['ground_truth_intermediates'][test_idx] = test_data[test_idx.item()].intermediates[1:]
        info['prompts'][test_idx] = prompt
        info['ground_truth'][test_idx] = test_data[test_idx.item()].broken_down_parts.products
        info['predicted'][test_idx] = predicted
        
    np.save(str(results_dir / "cot_prediction_step_results.npy"), dict(info), allow_pickle=True)
        


