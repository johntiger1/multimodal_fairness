#!/bin/bash
source activate hurtfulwords
RUN_NAME="60-preproc-decomp-50k"
printf '\033]2;%s\033\\' "$RUN_NAME"
srun --mem=60g -c 4 -p gpu --gres=gpu:1 --unbuffered  python MortalityMain.py --run_name="$RUN_NAME"
