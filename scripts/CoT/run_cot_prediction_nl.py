# File: run_baseline.py
# File Created: Friday, 9th June 2023 3:26:23 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Thursday, 27th July 2023 9:51:27 am
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Runs baseline Few Shot test.

from chempe.data.warnings import disable_warnings
from chempe.models.llm import generate_response_by_gpt
from chempe.test_management.results import TestManager, Results, ResultVariants


import openai
import argparse
import tqdm
import os


def create_prompt(test, examples):
    
    prompt = """You are an expert chemist. Your task is to predict the resulting product in this heterolytic linear electron flow topology chemical reaction given the reactants, several examples including intermediate steps, and your experienced reaction prediction knowledge. Heterolytic linear electron flow topology involves pairs of electrons, and each intermediate electron transfer step alternates between remove bonds and add bonds. Additionally, there are some rules to follow.
1. Strictly follow the given format.
2. Numbers immediately following : represent the atom mapping.
3. A . is used to distinguish between multiple molecules in the SMILES strings.
4. The resulting product must be be chemically reasonable and valid.
"""

    def find_am(am, smiles):
        idx = smiles.find(f':{am}')
        if idx == -1:
            raise Exception("No AM found")
        start_idx = idx - 1
        end_idx = idx + 1
        while True:
            if smiles[start_idx] == '[':
                break
            start_idx -= 1
        while True:
            if smiles[end_idx] == ']':
                break
            end_idx += 1
        return smiles[start_idx:end_idx+1]
    
    for idx, e in enumerate(examples):

        prompt += f"\nReactants: {e.intermediates[1]}\n"
        
        # invert map
        inverted_map = {}
        for am, index in e.am_to_idx_map.items():
            inverted_map[index] = am
        path = []
        for idx2 in e.ordered_indx_path:
            path.append(inverted_map[idx2])    
        
        state = 'remove'
        reasoning = ''
        for iidx, intermediate in enumerate(e.intermediates[2:], start=2): # skip first, cause always the same
            path_idx = iidx - 2
            if state == 'remove':
                # remove state
                state = 'add'
                if path[path_idx] == path[path_idx + 1]:
                    reasoning += f"A self bond removal is performed, changing {find_am(path[path_idx], e.intermediates[iidx - 1])} to {find_am(path[path_idx + 1], intermediate)}. "
                else:
                    reasoning += f"A bond removal is performed between {find_am(path[path_idx], e.intermediates[iidx - 1])} and {find_am(path[path_idx + 1], e.intermediates[iidx - 1])}. "
                    
            elif state == 'add':
                state = 'remove'
                reasoning = f"A bond is added between {find_am(path[path_idx], e.intermediates[iidx - 1])} and {find_am(path[path_idx + 1], e.intermediates[iidx - 1])}. "
                
            
            reasoning += f"The next intermediate is {intermediate}. "
        prompt += f"Products: {reasoning} The final intermediate is sanitized and the SMILES string is canonized. The final answer is {e.broken_down_parts.products}\n"
    prompt += f"\nReactants: {test.intermediates[1]}\n"
    prompt += "Products:"
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
                      filters=['nl'],
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
        
        print(predicted)
        rez = predicted[0].split("The final answer is ")[-1].strip()


        results.store(test_indice=test_idx,
                      prompt=prompt,
                      gt=test_data.broken_down_parts.products,
                      predicted=[rez],
                      icl_indices=TM.icl_indices[idx])
        
    results.save()

