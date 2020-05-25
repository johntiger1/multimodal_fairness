'''

From the predictions.csv which are generated, we will load everything in and check

'''
import pandas as pd
import os
import logging
from sklearn import metrics
logger = logging.getLogger(__name__)



def compute_auc(predictions_path: str=None):
    # pd readcsv each of those, concat, then we pass in the data to AUCROC
    #
    all_preds_df = pd.DataFrame()
    for root, dirs, files in os.walk(predictions_path):
        for file in files:
            if str(file).startswith("predictions_"):
                logger.critical(f"{file}\n")
                preds = pd.read_csv(os.path.join(root,file))
                all_preds_df = pd.concat((all_preds_df, preds), axis=0)

    all_preds_df["predictions"] = all_preds_df.apply(lambda row:  1 if row["probs_1"] > row["probs_0" ] else 0, axis=1)
    print(metrics.roc_auc_score(all_preds_df["label"], all_preds_df["probs_1"] ))

    # computed a different way: first, find the p/r curves, then find the area under it

    fpr,tpr, _ =metrics.roc_curve(all_preds_df["label"], all_preds_df["probs_1"] , 1)
    print(metrics.auc(fpr, tpr))
    print(len(all_preds_df))
    print(all_preds_df)

compute_auc(predictions_path="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/src/models/new_allen_nlp/experiments/43-fixed")