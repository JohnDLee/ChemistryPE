# File: run_baseline.py
# File Created: Friday, 9th June 2023 3:26:23 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Wednesday, 5th July 2023 7:19:30 pm
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
    
    prompt = """You are an expert chemist. Your task is to predict the next intermediate molecules in the electron transfer process given the reactants and reagents, several examples, and your experienced reaction prediction knowledge. There are some rules to follow.
1. Strictly follow the given format, and only respond with an atom mapped SMILES string.
2. Numbers immediately following : represent the atom mapping.
3. A . is used to distinguish between multiple molecules in the SMILES strings.
4. If no reagents exist, it is left blank.
5. The resulting intermediates must be be chemically reasonable and valid.
"""
    for idx, e in enumerate(examples):
        prompt += f"Reactants: {e.broken_down_parts.reactants}\n"
        prompt += f"Reagents: {e.broken_down_parts.reagents}\n"
        prompt += f"Intermediate: {e.intermediates[0]}\n"
    prompt += f"Reactants: {test.broken_down_parts.reactants}\n"
    prompt += f"Reagents: {test.broken_down_parts.reagents}\n"
    prompt += "Intermediate:"
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
    n = int(os.environ['NSAMPLES'])
    test_indices = np.load(os.environ['TEST_IDX'], allow_pickle=True)[:n]

    # set ICL samples
    k = int(os.environ['KICL'])
    
    # select model
    model = ModelVariants.GPT4 if args.use_gpt4 else ModelVariants.GPT3_5
    print(f"Using model {model}")
    
    # init openai
    openai.api_key = os.environ['OPEN_AI_KEY'] # loaded as environment variable
    
    # storage dir 
    results_dir = Path(os.environ['RESULTS_DIR']) / "step1_prediction_sc" / model.value
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # save information
    info = defaultdict(dict)
    # indices, prompts, answers
    # {test: [train]}, {test: prompt}, {test:(ybar, y)}
    
    # run through openai api
    for test_idx in tqdm.tqdm(test_indices, desc="Test Samples", ):
        train_indices = np.random.randint(0, len(train_data), k)
        prompt = create_prompt(test_data[test_idx.item()], train_data[train_indices])
        predicted = generate_response_by_gpt(prompt=prompt,
                                             model_engine=model,
                                             temperature=float(os.environ['SC_TEMP']),
                                             n=int(os.environ['SC_PREDS']))
        # print(prompt)

        info['icl_indices'][test_idx] = train_indices
        info['prompts'][test_idx] = prompt
        info['ground_truth'][test_idx] = test_data[test_idx.item()].intermediates[0]
        info['predicted'][test_idx] = predicted
        
    np.save(str(results_dir / "step1_prediction_sc_results.npy"), dict(info), allow_pickle=True)
        


