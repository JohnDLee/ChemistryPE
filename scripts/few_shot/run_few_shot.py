# File: run_baseline.py
# File Created: Friday, 9th June 2023 3:26:23 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Friday, 9th June 2023 4:31:03 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Runs baseline Few Shot test.

from chempe.data.lef_uspto import LEF_USPTO, DataVariants
from chempe.data.transforms import TransformStrToBrokenDownParts
from chempe.data.warnings import disable_warnings
import numpy as np

def create_prompt(test, examples):
    
    prompt = """You are an expert chemist. Your task is to predict the resulting product given the reactants and reagents and your experienced reaction prediction knowledge. There are some rules to follow.
1. Strictly follow the given format, and only respond with an atom mapped SMILES string and nothing else.
2. Numbers immediately following : are an atom mapping.
3. Provided reactants are split by a .
4. Provided reagents are split by a .
5. If no reagents exist, it is left blank.
6. The resulting product must be be chemically reasonable and valid.
"""
    for e in examples:
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
    
    # create a test prompt
    test_idx = 1
    k = 20
    train_idx = np.random.randint(0, len(train_data), k)
    print(create_prompt(test_data[test_idx], train_data[train_idx]))
    print(f"True Product: {test_data[test_idx].products}")

