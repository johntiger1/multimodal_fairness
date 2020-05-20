#!/bin/bash
source activate hurtfulwords
srun --mem=12g -p gpu --gres=gpu:1 --unbuffered  python MortalityModelDemo.py
