#!/bin/bash
source activate hurtfulwords
SER_DIR="experiments/20-config"
rm -rf $SER_DIR
srun --mem=30g -c 4 -p gpu --gres=gpu:1 --unbuffered  allennlp train MortalityClassifier.jsonnet -s $SER_DIR --include-package MortalityClassifier

