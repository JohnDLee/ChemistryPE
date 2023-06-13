# File: run_baseline.py
# File Created: Friday, 9th June 2023 3:26:23 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Monday, 12th June 2023 5:42:02 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Runs baseline Few Shot test.

from chempe.data.lef_uspto import LEF_USPTO, DataVariants
from chempe.data.transforms import TransformStrToBrokenDownParts
from chempe.data.warnings import disable_warnings
from chempe.models.llm import generate_response_by_gpt35

import numpy as np
import openai
import os

from pathlib import Path
from collections import defaultdict

def create_prompt(test, examples):
    
    prompt = """You are an expert chemist. Your task is to predict the resulting product given the reactants and reagents, several examples, and your experienced reaction prediction knowledge. There are some rules to follow.
1. Strictly follow the given format, and only respond with an atom mapped SMILES string.
2. Numbers immediately following : represent the atom mapping.
3. A . is used to distinguish between multiple reactant, reagent, and product SMILES strings.
4. If no reagents exist, it is left blank.
5. The resulting product must be be chemically reasonable and valid.
"""
    for idx, e in enumerate(examples):
        prompt += f"Reactants: {e.reactants}\n"
        prompt += f"Reagents: {e.reagents}\n"
        prompt += f"Product: {e.products}\n"
    prompt += f"Reactants: {test.reactants}\n"
    prompt += f"Reagents: {test.reagents}\n"
    prompt += "Product:"
    return prompt

if __name__ == '__main__':
    # disable rdkit warnings
    disable_warnings()
    
    # retrieve train/test data
    train_data = LEF_USPTO(DataVariants.TRAIN, transforms=[TransformStrToBrokenDownParts()])
    test_data = LEF_USPTO(DataVariants.TEST, transforms=[TransformStrToBrokenDownParts()])
    
    # select n test prompts
    n = 10
    test_indices = np.load(str(test_data.data_path.parent / "test_indices.npy"), allow_pickle=True)[:n]

    # set ICL samples
    k = 8
    
    # init openai
    openai.api_key = os.environ['OPEN_AI_KEY'] # loaded as environment variable
    
    # storage dir 
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # save information
    info = defaultdict(dict)
    # indices, prompts, answers
    # {test: [train]}, {test: prompt}, {test:(ybar, y)}
    
    # run through openai api
    for test_idx in test_indices:
        print(f"Testing {test_idx}")
        train_indices = np.random.randint(0, len(train_data), k)
        prompt = create_prompt(test_data[test_idx.item()], train_data[train_indices])
        
        predicted = generate_response_by_gpt35(prompt, temperature=0, n = 1)
        # print(prompt)

        info['indices'][test_idx] = train_indices
        info['prompts'][test_idx] = prompt
        info['answers'][test_idx] = (predicted[0], test_data[test_idx.item()].products)
    
    np.save(str(results_dir / "few_shot_results.npy"), dict(info), allow_pickle=True)
        


