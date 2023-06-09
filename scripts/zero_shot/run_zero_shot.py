# File: run_baseline.py
# File Created: Friday, 9th June 2023 3:26:23 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Friday, 9th June 2023 4:31:14 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Runs baseline Zero Shot test.

from chempe.data.lef_uspto import LEF_USPTO, DataVariants
from chempe.data.transforms import TransformStrToBrokenDownParts
from chempe.data.warnings import disable_warnings

def create_prompt(reactants, reagents):
    
    prompt = """You are an expert chemist. Your task is to predict the resulting product given the reactants and reagents and your experienced reaction prediction knowledge. There are some rules to follow.
1. Strictly follow the given format, and only respond with an atom mapped SMILES string and nothing else.
2. Numbers immediately following : are an atom mapping.
3. Provided reactants are split by a .
4. Provided reagents are split by a .
5. If no reagents exist, it is left blank.
6. The resulting product must be be chemically reasonable and valid.
"""
    prompt += f"Reactants: {reactants}\n"
    prompt += f"Reagents: {reagents}\n"
    prompt += f"Product:"
    return prompt

if __name__ == '__main__':
    # disable rdkit warnings
    disable_warnings()
    
    # retrieve test data
    test_data = LEF_USPTO(DataVariants.TEST, transforms=[TransformStrToBrokenDownParts()])
    
    # create a test prompt
    test_idx = 1
    print(create_prompt(test_data[test_idx].reactants, test_data[test_idx].reagents))
    print(f"True Product: {test_data[test_idx].products}")

