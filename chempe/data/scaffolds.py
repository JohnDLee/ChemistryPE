# File: scaffolds.py
# File Created: Friday, 21st July 2023 9:17:00 am
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Friday, 21st July 2023 10:11:16 am
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Methods to compute scaffolds for data

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from .warnings import disable_warnings, enable_warnings
from tqdm import tqdm

def get_scaffold_fp(smiles):
    """Gets scaffold fingerprints for SMILES x"""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_fp = rdMolDescriptors.GetMorganFingerprint(scaffold, 2)
    return scaffold_fp


def top_n_scaffold_similar_molecules(target_smiles, molecule_scaffold_list, n=5):
    """Use Tanimoto Similarity based on Fingerprints to find the top n similar molecules.
    Returns the index of the n molecules."""
    
    target_fp = get_scaffold_fp(target_smiles)

    similarities = []

    for idx, scaffold_fp in enumerate(molecule_scaffold_list):
        try:
            # compute tanimoto similarity
            tanimoto_similarity = DataStructs.TanimotoSimilarity(target_fp, scaffold_fp)
            similarities.append((idx, tanimoto_similarity))
        except Exception as e:
            print(e)
            continue

    # sort and get topn
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_n_similar_molecules = similarities[:n]
    
    return [i[0] for i in top_n_similar_molecules]
