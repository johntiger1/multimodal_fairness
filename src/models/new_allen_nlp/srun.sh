#!/bin/bash
source activate hurtfulwords
srun --mem=3g -p gpu --gres=gpu:1 --unbuffered  python ClassifyModelDemo.py
