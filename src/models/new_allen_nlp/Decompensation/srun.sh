#!/bin/bash
source activate hurtfulwords
srun --mem=30g -c 4 -p gpu --gres=gpu:1 --unbuffered  python DecompensationMain.py
