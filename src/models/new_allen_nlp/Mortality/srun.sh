#!/bin/bash
source activate hurtfulwords
RUN_NAME="55-named_srun"
printf '\033]2;%s\033\\' $RUN_NAME
srun --mem=15g -c 8 -p gpu --gres=gpu:1 --unbuffered  python MortalityMain.py --run_name=$RUN_NAME
