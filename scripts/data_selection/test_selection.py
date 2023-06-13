# File: test_selection.py
# File Created: Monday, 12th June 2023 5:31:15 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Monday, 12th June 2023 5:41:10 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Selects a reduced test set. (Or rather saves a randomized list of indices.)


from chempe.data.lef_uspto import LEF_USPTO
from chempe.data.lef_uspto import DataVariants

import numpy as np

if __name__ == '__main__':
    
    # load test data
    test_data = LEF_USPTO(DataVariants.TEST)
    
    # shuffle test data
    test_indices = np.array(range(0, len(test_data)))
    np.random.shuffle(test_indices)
    
    # save
    np.save(str(test_data.data_path.parent / "test_indices.npy"), test_indices, allow_pickle=True)