#!/bin/bash

MYVAR=/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/physionet.org/files/mimiciii/1.4
echo This script is used to also get textual data
python -m mimic3benchmark.scripts.extract_subjects $MYVAR mm_data/root/ --test