# File: lef_consistency_errors.py
# File Created: Friday, 9th June 2023 2:36:54 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Friday, 9th June 2023 2:37:25 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Lef errors. Taken from https://github.com/john-bradshaw/electro/blob/master/rxn_steps/data/lef_consistency_errors.py.





class NonLinearTopologyException(RuntimeError):
    """
    The atoms can be lined up end to end on the path
    """
    pass


class NotAddingAndRemovingError(RuntimeError):
    """
   The actions alternate between add and remove.
    """
    pass


class InconsistentActionError(RuntimeError):
    """
   The final molecule created by editing the reactants according to the action path
    """
    pass
