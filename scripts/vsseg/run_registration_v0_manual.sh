#!/usr/bin/env bash
set -eo pipefail

REPO=/NAS3/lbliao/Code-138/pytorch-CycleGAN-and-pix2pix
PYTHON=/data12/jing/anaconda3/envs/DHR/bin/python
ASLIDE=/NAS3/lbliao/Code-138/aslide

export PYTHONPATH="/NAS3/lbliao/Code-138:$ASLIDE:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/local/lib/aslide-lib/lib:$LD_LIBRARY_PATH"

cd "$REPO"
exec "$PYTHON" scripts/vsseg/registration_v0_manual.py "$@"
