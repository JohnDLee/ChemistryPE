# File: transforms.py
# File Created: Thursday, 8th June 2023 4:18:49 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Wednesday, 5th July 2023 6:47:10 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Transformations to extract electron paths. Taken from https://github.com/john-bradshaw/electro/blob/master/rxn_steps/data/transforms.py. Certain functions pertaining to adjacency lists are removed

import typing
import itertools

import numpy as np
from rdkit import Chem

from .rdkit_ops import rdkit_reaction_ops
from .rdkit_ops import rdkit_general_ops
# from .rdkit_ops import rdkit_featurization_ops
from .rdkit_ops import chem_details
from . import lef_consistency_errors
from . import action_list_funcs
# from . import mask_creators
# from . import graph_ds
# from ..misc import data_types

"""
 ==== Datastructures from the different transforms ====
"""

class BrokenDownParts(typing.NamedTuple):
    reactants: str  # AM-SMILES
    reagents: str  # AM-SMILES
    products: str  # AM-SMILES
    ordered_am_path: typing.List[int]  # should include self-bond break at start if applicable by repeated action.


class RdKitIntermediates(typing.NamedTuple):
    broken_down_parts: BrokenDownParts
    intermediates: typing.List[Chem.Mol]
    reagents: Chem.Mol
    ordered_indx_path: typing.List[int]
    am_to_idx_map: typing.Mapping[int, int]
    

"""
 ==== Transforms between the different datastructures ====
"""
    
class TransformStrToBrokenDownParts:
    """
    (A) Creates path and assigns order see Fig 3 of the paper.
    (B) Breaks out the reagents,

    It runs the following consistency checks:
    C1. The atoms can be lined up end to end on the path
    """
    def __call__(self, input_: typing.Tuple[str, str, str]) -> BrokenDownParts:
        reactants, products, bond_changes = input_

        # A.2 We first create the path (checks for C1.)
        change_list = bond_changes.split(';')
        atom_pairs = np.array([c.split('-') for c in change_list]).astype(int)
        unordered_electron_path_am = action_list_funcs.actions_am_from_pairs(atom_pairs, consistency_check=True)

        # B Using these actions we can now work out what is a reagents and split these out.
        reactants, reagents, products = rdkit_reaction_ops.split_reagents_out_from_reactants_and_products(
            reactants, products, unordered_electron_path_am)

        # A.3 We can now order the electron path.
        reactant_mol = rdkit_general_ops.get_molecule(reactants, kekulize=False)
        product_mol = rdkit_general_ops.get_molecule(products, kekulize=False)
        ordered_electon_path_am = action_list_funcs.order_actions_am(unordered_electron_path_am, reactant_mol, product_mol)

        # A.3b Work out whether to add add a self-bond remove at the start
        # (we need to start with a remove action to pick up a pair of electrons)
        first_bond_in_reactants = rdkit_general_ops.get_bond_double_between_atom_mapped_atoms(reactant_mol, ordered_electon_path_am[0],
                                                                                     ordered_electon_path_am[1])
        first_bond_in_products = rdkit_general_ops.get_bond_double_between_atom_mapped_atoms(product_mol, ordered_electon_path_am[0],
                                                                                    ordered_electon_path_am[1])
        starts_already_with_remove_bond = first_bond_in_reactants - first_bond_in_products > 0
        if not starts_already_with_remove_bond:
            ordered_electon_path_am = [ordered_electon_path_am[0]] + ordered_electon_path_am

        # We can now create the o/p data-structure
        op = BrokenDownParts(reactants, reagents, products, ordered_electon_path_am)
        return op


class TransformToRdKitIntermediates:
    """
    This creates a series of intermediate molecules showing the molecule state at each stage of the reaction path.
    Effectively this is panel 4 of Fig3 of the the paper.

    It runs the following consistency checks:
    C2. The bond differences between the reactants and products along the path vary by +- one
    C3. The final molecule created by editing the reactants should be consistent with reported product.
         However, it can contain more information, eg minor products.
        we therefore check consistency of "super molecule" formed by changing actions versus the "sub molecule",
         which is the product given in the dataset. The super molecule should have disconnected extra molecules but
         the major product should show up in it.
    """
    def __call__(self, reaction_parts: BrokenDownParts) -> RdKitIntermediates:
        # 1. Work out the number of add/remove steps.
        # We always start on a remove bond (can be remove self bond though) and go in intermediate add/remove steps)
        # Therefore the add remove steps is as follows:
        action_types = [bond_change for bond_change, _ in
                 zip(
                     itertools.cycle([chem_details.ElectronMode.REMOVE, chem_details.ElectronMode.ADD]),
                     range(len(reaction_parts.ordered_am_path) -1)
                 )]

        # 2. Form the intermediate states
        reactant_mol = rdkit_general_ops.get_molecule(reaction_parts.reactants, kekulize=True)
        reactant_atom_map_to_idx_map = rdkit_general_ops.create_atom_map_indcs_map(reactant_mol)
        intermediates = [reactant_mol, reactant_mol]  # twice as after picked up half a pair nothing has happened.
        action_pairs = zip(reaction_parts.ordered_am_path[:-1], reaction_parts.ordered_am_path[1:])
        for step_mode, (start_atom_am, next_atom_am) in zip(action_types, action_pairs):
            # Nb note we do not change any Hydrogen numbers on the first and last step -- these may change in practice
            # eg gaining H from water in solution, however, we do not represent Hydrogens in our graph structure,
            # apart from in the features.
            prev_mol = intermediates[-1]
            start_atom_idx = reactant_atom_map_to_idx_map[start_atom_am]
            next_atom_idx = reactant_atom_map_to_idx_map[next_atom_am]
            if start_atom_idx == next_atom_idx:
                # then a self bond removal:
                intermed = rdkit_reaction_ops.change_mol_atom(prev_mol, step_mode, start_atom_idx)
            else:
                intermed = rdkit_reaction_ops.change_mol_bond(prev_mol, step_mode, (start_atom_idx, next_atom_idx))
            intermediates.append(intermed)

        # 3. Form the reagent molecule
        reagent_mol = rdkit_general_ops.get_molecule(reaction_parts.reagents, kekulize=True)

        # 4. Consistency checks.
        product_mol = rdkit_general_ops.get_molecule(reaction_parts.products)
        # we first check C2:
        if not rdkit_reaction_ops.is_it_alternating_add_and_remove_steps(reactant_mol, product_mol,
                                                                         reaction_parts.ordered_am_path):
            raise lef_consistency_errors.NotAddingAndRemovingError

        if not rdkit_reaction_ops.is_sub_mol_consistent_with_super(product_mol, intermediates[-1]):
            raise lef_consistency_errors.InconsistentActionError(f"Inconsistent action error for molecule:"
                                                                 f" {[str(part) for part in reaction_parts]}")

        # 5. Switch ordered path to refer to atoms indices rather than atom mapped number.
        ordered_index_path = [reactant_atom_map_to_idx_map[am] for am in reaction_parts.ordered_am_path]

        # 6. Create the final output
        # convert intermediates to smiles
        for idx in range(len(intermediates)):
            intermediates[idx] = Chem.MolToSmiles(intermediates[idx])
        op = RdKitIntermediates(reaction_parts, intermediates, reagent_mol, ordered_index_path, reactant_atom_map_to_idx_map)
        return op
