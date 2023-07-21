# File Created: Friday, 9th June 2023 3:26:23 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Thursday, 6th July 2023 9:46:27 pm
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
    

    
    # disable rdkit warnings
    disable_warnings()
    
    # retrieve train/test data
    train_data = LEF_USPTO(DataVariants.TRAIN, transforms=[TransformStrToBrokenDownParts(), TransformToRdKitIntermediates()])
    test_data = LEF_USPTO(DataVariants.TEST, transforms=[TransformStrToBrokenDownParts(), TransformToRdKitIntermediates()])
    
    with open("samples.txt", 'w') as sf:
        
        for i in range(5):
            sf.write(test_data[i].broken_down_parts.reactants + '\n')
            for j in test_data[i].intermediates:
                sf.write(j + '\n')
            sf.write(test_data[i].broken_down_parts.products + '\n\n')
            
        