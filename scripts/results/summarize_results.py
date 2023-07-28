# File: calculate_results.py
# File Created: Wednesday, 26th July 2023 10:33:31 am
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Thursday, 27th July 2023 7:38:14 am
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Looks into results dir and computes some statistics for for all data (updates results if the .npz file is newer than the statistics files)

from chempe.test_management.results import Results, ResultVariants
from chempe.data.warnings import disable_warnings

import os
from pathlib import Path
import argparse
import tqdm
import json
from collections import Counter

from rdkit import Chem

def CanonSmiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=True))
    except Exception as e:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))


def collect_unsummarized_results(use_all = False):
    """Collects all dirpaths that contain results that need to be summarized"""
    collected_dirpaths = []
    
    # for each dirpath
    for (dirpath, subdirpaths, filepaths) in os.walk(str(Path(args.root) / 'results')):
        
        # if 1 or more file exists (check if newer)
        if len(filepaths) >= 1:
            result_time = 0
            other_time = 0

            for fpath in filepaths:
                fpath = os.path.join(dirpath, fpath)
                if other_time and result_time: break # end if 2 different times were found
                
                if Path(fpath).name == 'results.npz':
                    # results modification time
                    result_time = os.stat(fpath).st_mtime
                else:
                    other_time = os.stat(fpath).st_mtime
            
            # if there is no results previously computed, other time is 0
            if result_time > other_time: # results are newer than the other files
                collected_dirpaths.append(dirpath)
            else:
                if use_all:
                    collected_dirpaths.append(dirpath)
    return collected_dirpaths
        
def summarize_results(root, results_dirpath, print_=False):
    
    results = Results(proj_root=root,
                      header=None,
                      model=None,
                      scaffold=None,
                      filters=None,
                      variant=None,
                      save_dir=results_dirpath)
    # load results
    results.load()
    
    # for each variant, use a different computation method
    if results.variant == ResultVariants.VANILLA:
        summary, comparisons = summ_vanilla(results)
    elif results.variant == ResultVariants.SC:
        summary, comparisons = summ_sc(results)
    #! May add more variants in the future
    else:
        print(f"Results at {results_dirpath} was not labeled with a variant. Skipping.")
        return
    
    # save summary
    sum_path = results.save_dir / 'summary.json'
    with sum_path.open('w') as fp:
        json.dump(summary, fp,indent=0)
        
    # print if desired
    if print_:
        print(results_dirpath +':')
        for key, value in summary.items():
            print(f"\t{key}: {value}")

    # save comparison
    save_comparison(results, comparisons)
    
    # sample prompt
    prompt_sample = results.save_dir / 'sample_prompt.txt'
    with prompt_sample.open("w") as fp:
        fp.write(results.prompts[0])
    
def save_comparison(results, comparisons):
    comp_path = results.save_dir / 'comparisons.txt'
    with comp_path.open('w') as fp:
        # vanilla
        if results.variant == ResultVariants.VANILLA:
            for validity, pred, gt in comparisons:
                fp.write(validity + '\n')
                fp.write("Predicted:\n")
                for p in pred:
                    fp.write("\t" + p + '\n')
                fp.write("GT:\n")
                for g in gt:
                    fp.write("\t" + g + '\n')
                fp.write('\n')
        # self consistency
        elif results.variant == ResultVariants.SC:
            for validity, pred, gt in comparisons:
                fp.write(validity + '\n')
                fp.write("Predicted:\n")
                for idx, p in enumerate(pred):
                    fp.write(f"\t{idx + 1} - Count ({p[1]}):\n")
                    for smiles in p[0].split("."):
                        fp.write("\t\t" + smiles + '\n')
                fp.write("GT:\n")
                for g in gt:
                    fp.write("\t" + g + '\n')
                fp.write('\n')

def summ_sc(results: Results):
    """Use consistency"""
    
    top1 = 0
    topn = 0
    invalid = 0
    n = len(results.test_indices)
    comparisons = []
    
    def count_preds(predicted):
        """Counts the number of valid predicted smiles"""
        canon_pred = []
        for p in predicted:
            try:
                canon_pred.append(CanonSmiles(p))
            except:
                pass
        #print(tmp)
        return Counter(canon_pred)
        

    for i in range(n):
        # canonical gt
        gt = set(CanonSmiles(results.gt[i]).split('.'))
        counter = count_preds(results.predicted[i])
        if len(counter) == 0:
            invalid += 1
            comparisons.append(("Invalid", Counter(results.predicted[i]).most_common(), list(gt)))
            continue

        # top 1
        pred = counter.most_common()
        
        top1_pred = set(pred[0][0].split("."))
        if gt.issubset(top1_pred):
            top1 += 1
            comparisons.append(("Correct (Top 1)",[pred[0]], list(gt)))
            continue
        
        # top n
        for idx, (smiles, count) in enumerate(pred):
            tmp_pred = set(smiles.split("."))
            if gt.issubset(tmp_pred):
                topn += 1
                comparisons.append((f"Correct (Top ({idx + 1}))", pred, list(gt)))
                break
        # not found at all
        else:
            comparisons.append(("Incorrect", pred, list(gt)))
            
            
        
    # correct topn
    topn += top1    
    
    return {"top1": top1,
            "topn": topn,
            "invalid": invalid,
            "n": n}, comparisons

# evaluate results
def summ_vanilla(results: Results):
    """Only 1 result"""
        
    top1 = 0
    partial1 = 0
    invalid = 0
    n = len(results.test_indices)
    
    comparisons = []

    for i in range(n):
        # canonical gt
        gt = set(CanonSmiles(results.gt[i]).split('.'))
        try:
            pred = set(CanonSmiles(results.predicted[i][0]).split("."))
            if gt.issubset(pred):
                top1 += 1
                comparisons.append(('Correct', list(pred), list(gt)))
            elif gt.intersection(pred):
                partial1 += 1
                comparisons.append(("Partial Correct", list(pred), list(gt)))
            else:
                comparisons.append(('Incorrect', list(pred), list(gt)))
        except Exception as e:
            # if failed to parse
            invalid += 1
            comparisons.append(('Invalid', results.predicted[i][0].split("."), list(gt)))
    
    return {'top1': top1,
            'partial1': partial1,
            'invalid': invalid,
            'n':n}, comparisons

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Attempts to use GPT to convert to kekulized form as step 1")
    parser.add_argument("-root", dest="root", default=".", help="Root Project Dir")
    parser.add_argument("-p", dest='print', action = 'store_true', default = False, help = 'print summary results')
    parser.add_argument("-all", action="store_true", default=False, help="Rerun all.")
    
    args = parser.parse_args()
    
    disable_warnings()
    
    collected_dirpaths = collect_unsummarized_results(use_all=args.all)
    
    for results_dirpath in tqdm.tqdm(collected_dirpaths, desc="Summarizing Results", total = len(collected_dirpaths)):
        summarize_results(args.root, results_dirpath, args.print)
    