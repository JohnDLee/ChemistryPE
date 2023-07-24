# File: run_baseline.py
# File Created: Friday, 9th June 2023 3:26:23 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Monday, 24th July 2023 8:48:40 am
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
    
    prompt = """You are an expert chemist. Your task is to predict the next intermediate molecule in the electron transfer process given the previous intermediate molecule, several examples, and your experienced reaction prediction knowledge. There are some rules to follow.
1. Strictly follow the given format, and only respond with an atom mapped SMILES string.
2. Numbers immediately following : represent the atom mapping.
3. A . is used to distinguish between multiple molecules in the SMILES strings.
4. If no reagents exist, it is left blank.
"""
#5. The resulting intermediates must be be chemically reasonable and valid.
    for idx, e in enumerate(examples):
        for iidx, intermediate in enumerate(e.intermediates[1:3]): # skip first, cause always the same
            prompt += f"Intermediate {iidx + 1}: {intermediate}\n"
    prompt += f"Intermediate 1: {test.intermediates[1]}\n"
    prompt += f"Intermediate 2:"
    
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
    icl_indices = np.load(os.environ['ICL_IDX'], allow_pickle=True)
    
    # select model
    model = ModelVariants.GPT4 if args.use_gpt4 else ModelVariants.GPT3_5
    print(f"Using model {model}")
    
    # init openai
    openai.api_key = os.environ['OPEN_AI_KEY'] # loaded as environment variable
    
    # storage dir 
    results_dir = Path(os.environ['RESULTS_DIR']) / "step2_prediction_int" / model.value
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
        predicted = generate_response_by_gpt(prompt=prompt,
                                             model_engine=model,
                                             temperature=float(os.environ['B_TEMP']),
                                             n=int(os.environ['B_PREDS']))
        if p:
            print(prompt)
            p = False

        info['icl_indices'][test_idx] = train_indices
        info['prompts'][test_idx] = prompt
        info['ground_truth'][test_idx] = test_data[test_idx.item()].intermediates[2]
        info['predicted'][test_idx] = predicted
        
    np.save(str(results_dir / "step2_prediction_int_results.npy"), dict(info), allow_pickle=True)
        

