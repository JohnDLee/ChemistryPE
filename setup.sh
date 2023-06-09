#!/usr/bin/env bash
# source this script to set up python paths

MAIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set up the Python Paths
export PYTHONPATH=${PYTHONPATH}:${MAIN_DIR}
export PYTHONPATH=${PYTHONPATH}:${MAIN_DIR}/scratch