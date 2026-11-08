#!/bin/bash

### must be used in interactive dashboard on oscar
### use in terminal with "bash run_monkey.sh"
### opens isaacgym UI and runs example "joint_monkey.py"

cd ~/CSCI1952A/isaacgym/python/examples || exit 1
export PYTHONPATH="$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec env -u LD_PRELOAD "$CONDA_PREFIX/bin/python" joint_monkey.py