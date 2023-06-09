# File: lef_uspto.py
# File Created: Thursday, 8th June 2023 3:32:41 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Friday, 9th June 2023 3:24:30 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Class for parsing LEF USPTO dataset in data folder

from pathlib import Path
import enum
import typing

class DataVariants(enum.Enum):
    TRAIN="filtered_train.txt"
    VAL="filtered_val.txt"
    TEST="filtered_test.txt"

class LEF_USPTO:
    """Loads the corresponding data variant of the lef_ustpo dataset into memory.
    """
    
    def __init__(self, variant: DataVariants, transforms = None):
        
        # Get main data folder
        data_path = Path(__file__).parent.parent.parent / "data" / "lef_uspto"
        if not data_path.exists():
            raise FileNotFoundError(str(data_path) + 'does not exist. Please unzip the provided lef_uspto.zip file.')
        
        # Load variant
        variant_path = data_path / variant.value
        with variant_path.open("r") as f:
            self.rxn_data = f.readlines()
        
        # transformations
        self.transforms = transforms
            
    def __getitem__(self, idx: typing.Union[int, typing.Iterable]):
        """Retrieves item

        Args:
            idx (typing.Union[int, typing.Iterable]): either an indice or an iterable of indices
        """
        if type(idx) == int:
            # if int is passed
            rxn = self.rxn_data[idx].rstrip()
            return self._transform(self._parse_txt(rxn))
        else:
            # some iterable is passed
            data = []
            for i in idx:
                data.append(self._transform(self._parse_txt(self.rxn_data[i].rstrip())))
            return data
    
    def _transform(self, tup):
        """Transformations if they exist
        """
        if self.transforms:
            for t in self.transforms:
                tup = t(tup)
        return tup

    def _parse_txt(self, rxn):
        """Parsing for the txt files
        """
        # Seperate reactants, products, and bond changes
        smiles, bond_changes = rxn.split()
        reactants, products = smiles.split(">>")
        return (reactants, products, bond_changes)
        
    def __len__(self) -> int:
        return len(self.rxn_data)
        
        