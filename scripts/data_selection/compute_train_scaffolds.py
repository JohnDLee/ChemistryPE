# File: compute_train_scaffolds.py
# File Created: Friday, 21st July 2023 9:12:25 am
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Friday, 21st July 2023 10:23:48 am
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Computes and saves the scaffold fingerprints for the training data


from chempe.data.scaffolds import get_scaffold_fp, top_n_scaffold_similar_molecules
from chempe.data.lef_uspto import LEF_USPTO, DataVariants
from chempe.data.warnings import disable_warnings

import numpy as np
import os
import tqdm


if __name__ == '__main__':
    
    disable_warnings()
    
    # get training data with no transform
    train_data = LEF_USPTO(DataVariants.TRAIN, [lambda x: x])
    
    fingerprints = []
    for sample_idx in tqdm.trange(0, len(train_data), desc="Computing Scaffold Fingerprints"):
        reactant_smiles = train_data[sample_idx][0]
        fingerprints.append(get_scaffold_fp(reactant_smiles))
    
    
    # get test data 
    test_data = LEF_USPTO(DataVariants.TEST, [lambda x: x])
    
    # top 20 similar
    n = 20
    similar_mols = []

    for test_idx in tqdm.trange(0, len(test_data), desc="Computing similar molecules for test data"):
        target_smiles = test_data[test_idx][0]
        similar_mols.append(top_n_scaffold_similar_molecules(target_smiles,fingerprints,n))
                
    
    np.save(os.path.join(os.environ['MAIN_DIR'], "data/test_ICL_indices.npy"), similar_mols, allow_pickle=True)
    
    
        

