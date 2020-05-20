#!/bin/bash
source activate hurtfulwords
srun --mem=15g -c 4 -p gpu --gres=gpu:1 --unbuffered  python MortalityModelDemo.py
