#!/bin/bash
# Activate the rtg132 conda environment for TripoSplat.
# Source this file before running any script:
#   source setup_env.sh

export PATH="/c/Users/pcl/.conda/envs/rtg132/Scripts:/c/Users/pcl/.conda/envs/rtg132/Library/bin:$PATH"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Verify
echo "Environment ready:"
echo "  Python: $(which python 2>/dev/null || echo '/c/Users/pcl/.conda/envs/rtg132/python.exe')"
echo "  PYTHONUTF8=$PYTHONUTF8"
echo "  ninja: $(which ninja 2>/dev/null || echo 'NOT FOUND - check PATH')"
