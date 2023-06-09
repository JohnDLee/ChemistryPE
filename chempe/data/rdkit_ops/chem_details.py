
# File: chem_details.py
# File Created: Friday, 9th June 2023 2:26:49 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Friday, 9th June 2023 2:31:47 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Chemical details. Taken from https://github.com/john-bradshaw/electro/tree/master/rxn_steps/data/rdkit_ops

"""
from RDKIT: https://github.com/rdkit/rdkit/blob/f4529c910e546af590c56eba01f96e9015c269a6/Code/GraphMol/atomic_data.cpp
"""
import enum

class ElectronMode(enum.Enum):
 ADD = 0
 REMOVE = 1



DEFAULT_ELEM_VALENCES = {'Ac': -1,
 'Ag': -1,
 'Al': 6,
 'Am': -1,
 'Ar': 0,
 'As': 3,
 'At': 1,
 'Au': -1,
 'B': 3,
 'Ba': 2,
 'Be': 2,
 'Bh': -1,
 'Bi': 3,
 'Bk': -1,
 'Br': 1,
 'C': 4,
 'Ca': 2,
 'Cd': -1,
 'Ce': -1,
 'Cf': -1,
 'Cl': 1,
 'Cm': -1,
 'Cn': -1,
 'Co': -1,
 'Cr': -1,
 'Cs': 1,
 'Cu': -1,
 'Db': -1,
 'Ds': -1,
 'Dy': -1,
 'Er': -1,
 'Es': -1,
 'Eu': -1,
 'F': 1,
 'Fe': -1,
 'Fm': -1,
 'Fr': 1,
 'Ga': 3,
 'Gd': -1,
 'Ge': 4,
 'H ': 1,
 'He': 0,
 'Hf': -1,
 'Hg': -1,
 'Ho': -1,
 'Hs': -1,
 'I': 1,
 'In': 3,
 'Ir': -1,
 'K': 1,
 'Kr': 0,
 'La': -1,
 'Li': 1,
 'Lr': -1,
 'Lu': -1,
 'Md': -1,
 'Mg': 2,
 'Mn': -1,
 'Mo': -1,
 'Mt': -1,
 'N': 3,
 'Na': 1,
 'Nb': -1,
 'Nd': -1,
 'Ne': 0,
 'Ni': -1,
 'No': -1,
 'Np': -1,
 'O': 2,
 'Os': -1,
 'P': 5,#3, # NB was 3 before but have some molecules S combining with 6 in Zinc dataset.
 'Pa': -1,
 'Pb': 4,
 'Pd': -1,
 'Pm': -1,
 'Po': 2,
 'Pr': -1,
 'Pt': -1,
 'Pu': -1,
 'Ra': 2,
 'Rb': 1,
 'Re': -1,
 'Rf': -1,
 'Rg': -1,
 'Rh': -1,
 'Rn': 0,
 'Ru': -1,
 'S': 6, #2, # NB was two before but have some molecules S combining with 6 in Zinc dataset.
 'Sb': 3,
 'Sc': -1,
 'Se': 2,
 'Sg': -1,
 'Si': 4,
 'Sm': -1,
 'Sn': 4,
 'Sr': 2,
 'Ta': -1,
 'Tb': -1,
 'Tc': -1,
 'Te': 2,
 'Th': -1,
 'Ti': -1,
 'Tl': 3,
 'Tm': -1,
 'U': -1,
 'V': -1,
 'W': -1,
 'Xe': 0,
 'Y': -1,
 'Yb': -1,
 'Zn': -1,
 'Zr': -1}


electroneg = {'Ag': 1.93, 'Al': 1.61, 'Ar': 3.98, # set to max to mean very unlikely
              'As': 2.18, 'Au': 2.54, 'B':  2.04,
              'Ba': 0.89, 'Be': 1.57, 'Bi': 2.02,
              'Br': 2.96, 'C':  2.55, 'Ca': 1.0,
              'Cd': 1.69, 'Ce': 1.12, 'Cl': 3.16,
              'Co': 1.88, 'Cr': 1.66, 'Cs': 0.79,
              'Cu': 1.90, 'Dy': 1.22, 'Eu': 3.98,
              'F':  3.98, 'Fe': 1.83, 'Ga': 1.81,
              'Ge': 2.01, 'H':  2.20, 'He': 3.98,
              'Hf': 1.3,  'Hg': 2.0,  'I':  2.66,
              'In': 1.78, 'Ir': 2.20, 'K':  0.82,
              'La': 1.10, 'Li': 0.98, 'Mg': 1.31,
              'Mn': 1.55, 'Mo': 2.16, 'N':  3.04,
              'Na': 0.93, 'Nd': 1.14, 'Ni': 1.91,
              'O':  3.44, 'Os': 2.20, 'P':  2.19,
              'Pb': 2.33, 'Pd': 2.20, 'Pr': 1.13,
              'Pt': 2.28, 'Pu': 1.28, 'Ra': 0.9,
              'Rb': 0.82, 'Re': 1.9,  'Rh': 2.28,
              'Rn': 3.98, 'Ru': 2.2,  'S':  2.58,
              'Sb': 2.05, 'Sc': 1.36, 'Se': 2.55,
              'Si': 1.90, 'Sm': 1.17, 'Sn': 1.96,
              'Sr': 0.95, 'Ta': 1.5,  'Tb': 3.98,
              'Tc': 1.9,  'Te': 2.1,  'Th': 1.3,
              'Ti': 1.54, 'Tl': 1.62, 'Tm': 1.25,
              'U':  1.38, 'V':  1.63, 'W':  2.36,
              'Xe': 2.6,  'Y':  1.22, 'Yb': 3.98,
              'Zn': 1.65, 'Zr': 1.33}