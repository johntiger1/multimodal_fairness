#!/bin/bash
rm -rf experiments/24
allennlp train MortalityBERTClassifier.jsonnet -s experiments/24 --include-package MortalityClassifier
