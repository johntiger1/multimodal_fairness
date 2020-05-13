#!/bin/bash
srun --mem=10g -p cpu --unbuffered python src/preprocessing/extract_notes.py