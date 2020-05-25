'''

From the predictions.csv which are generated, we will load everything in and check

'''
import pandas
import os
import logging
logger = logging.getLogger(__name__)

def compute_auc(predictions_path: str=None):
    # pd readcsv each of those, concat, then we pass in the data to AUCROC
    #
    for root, dirs, files in os.walk(predictions_path):
        for file in files:
            if str(file).startswith("predictions_"):
                logger.critical(f"{file}\n")


compute_auc(predictions_path="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/src/models/new_allen_nlp/experiments/50-mort-balanced-10k")