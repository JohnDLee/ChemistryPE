###########
# General #
###########

# dir
export MAIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export RESULTS_DIR="$MAIN_DIR/results"

# test indices
export TEST_IDX="$MAIN_DIR/data/test_indices.npy"
export ICL_IDX="$MAIN_DIR/data/test_ICL_indices.npy"

# number of test samples
export NSAMPLES=20

#######
# ICL #
#######

# numer of ICL samples
export KICL=5

############
# Baseline #
############

export B_TEMP=0
export B_PREDS=1

####################
# Self-Consistency #
####################

# temperature
export SC_TEMP=0.5
export SC_PREDS=20