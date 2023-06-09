# File: warnings.py
# File Created: Friday, 9th June 2023 3:21:01 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Friday, 9th June 2023 3:22:20 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Disables rdkit warnings.

from rdkit import RDLogger

def disable_warnings():
    RDLogger.DisableLog("rdApp.*")
    
def enable_warnings():
    RDLogger.EnableLog("rdApp.*")