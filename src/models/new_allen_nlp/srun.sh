#!/bin/bash
source activate hurtfulwords
printf '\033]2;%s\033\\' 'my reason for running'
srun --mem=30g -c 4 -p gpu --gres=gpu:1 --unbuffered  python MortalityModelDemo.py
