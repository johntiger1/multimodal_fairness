# NOTE: RUN in python3 to get more sklearn hyperparameters
from __future__ import absolute_import


import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from mimic3models import metrics

# Command to run on mortality data (may need to change paths to John's data):
# python3 -um ensemblemodels.logistic_regression --mode M --str_path ./mimic3models/in_hospital_mortality/train_predictions/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt.csv  --ustr_path ./IanFairnessHackery/john_results/train_final_preds_mort_id_ep_fmt.csv --test_str_path ./mimic3models/in_hospital_mortality/test_predictions/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt.csv  --test_ustr_path ./IanFairnessHackery/john_results/final_preds_mort_id_ep_fmt.csv --outdir ./ensemblemodels/mortality/test

# Command to run on train mortality data
# python3 -um ensemblemodels.logistic_regression --mode M --str_path ./mimic3models/in_hospital_mortality/train_predictions/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt.csv  --ustr_path ./IanFairnessHackery/john_results/train_final_preds_mort_id_ep_fmt.csv --test_str_path ./mimic3models/in_hospital_mortality/train_predictions/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt.csv  --test_ustr_path ./IanFairnessHackery/john_results/train_final_preds_mort_id_ep_fmt.csv --outdir ./ensemblemodels/mortality/train

# Command to run on validation mortality data
# python3 -um ensemblemodels.logistic_regression --mode M --str_path ./mimic3models/in_hospital_mortality/train_predictions/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt.csv  --ustr_path ./IanFairnessHackery/john_results/train_final_preds_mort_id_ep_fmt.csv --test_str_path ./mimic3models/in_hospital_mortality/val_predictions/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt.csv  --test_ustr_path ./IanFairnessHackery/john_results/train_final_preds_mort_id_ep_fmt.csv --outdir ./ensemblemodels/mortality/val

#set args for model HERE so that args can go into save file name
MODEL_ARGS={'penalty': 'l1', 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 0.25}


def load_data(mode, str_path, unstr_path):
    unstructured = pd.read_csv(unstr_path)
    structured = pd.read_csv(str_path)

    # Determine which columns form the key of the data based on the task, and rename columns to be consistent
    if (mode in ['M', 'P']):
        key_cols = ["patient_id", "episode", "label"]
        unstructured.rename(columns={unstructured.columns[0]: "patient_id",
                                     unstructured.columns[1]: "episode",
                                     unstructured.columns[2]: "unstr_prediction",
                                     unstructured.columns[3]: "label"}, inplace=True)
        structured.rename(columns={structured.columns[0]: "patient_id",
                                   structured.columns[1]: "episode",
                                   structured.columns[2]: "str_prediction",
                                   structured.columns[3]: "label"}, inplace=True)
    else:
        key_cols = ["patient_id", "episode", "time", "label"]
        unstructured.rename(columns={unstructured.columns[0]: "patient_id",
                                     unstructured.columns[1]: "episode",
                                     unstructured.columns[2]: "time",
                                     unstructured.columns[3]: "unstr_prediction",
                                     unstructured.columns[4]: "label"}, inplace=True)
        structured.rename(columns={structured.columns[0]: "patient_id",
                                   structured.columns[1]: "episode",
                                   structured.columns[2]: "time",
                                   structured.columns[3]: "str_prediction",
                                   structured.columns[4]: "label"}, inplace=True)

    print("Structured")
    print(structured)
    print("Unstructured")
    print(unstructured)

    print("Joined")
    merged_results = structured.merge(unstructured, how="inner", on=key_cols, validate="one_to_one")
    print(merged_results)

    # Sanity check
    assert(not merged_results.isnull().values.any())
    return merged_results

def load_fit_save(mode, str_path, ustr_path, test_str_path, test_ustr_path, outdir, out_filename_prefix = None):
    print("LOAD TRAIN DATA")
    train = load_data(mode, str_path, ustr_path)
    print("=========================\n")
    print("LOAD TEST DATA")
    test = train = load_data(mode, test_str_path, test_ustr_path)
    print("=========================\n")

    Y = train.label.values
    X = np.stack((train.str_prediction.values, train.unstr_prediction.values), axis=1)
    print(X)
    print(Y)

    model = LogisticRegression(**MODEL_ARGS).fit(X, Y)
    print(model)
    print(model.classes_)

    # Get test results:
    outname1 = os.path.basename(test_str_path)
    outname2 = os.path.basename(test_ustr_path)
    assert (outname1.endswith(".csv"))
    assert (outname2.endswith(".csv"))

    if out_filename_prefix is None:
        outname = outname1[:-4] + '+' + outname2[:-4] + repr(MODEL_ARGS).replace(': ', '=') + "_id_ep_fmt.csv"
    else:
        outname = out_filename_prefix + repr(MODEL_ARGS).replace(': ', '=') + "_id_ep_fmt.csv"

    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass
    outpath = os.path.join(outdir, outname)

    with open(outpath, 'w') as fw:
        if mode in ["P", "M"]:
            fw.write("patient_id, episode, prediction, label\n")
        else:
            fw.write("patient_id, episode,time, prediction, label\n")

        test_Y = test.label.values
        test_X = np.stack((test.str_prediction.values, test.unstr_prediction.values), axis=1)
        preds = model.predict_proba(test_X)
        preds = preds.T[1]  # get just probabilities of class 1
        # print(preds)
        print("Model Arguments:")
        print(MODEL_ARGS)
        metrics.print_metrics_binary(test_Y, preds)

        if mode in ['P', 'M']:
            for id, ep, pred, label in zip(test.patient_id.values, test.episode.values, preds, test_Y):
                fw.write("{},{},{},{}\n".format(id, ep, pred, label))
        else:
            for id, ep, time, pred, label in zip(test.patient_id.values, test.episode.values, test.time.values, preds,
                                                 test_Y):
                fw.write("{},{},{},{},{}\n".format(id, ep, time, pred, label))

        return test_Y, preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ustr_path', required=True, type=str,
                        help="Training Path to .csv file of unstructured model results, for phenotype directory")
    parser.add_argument('--str_path', required=True, type=str,
                        help="Training Path to .csv file of structured model results, for phenotype directory")
    parser.add_argument('--test_ustr_path', required=True, type=str,
                        help="Test Path to .csv file of unstructured model results, for phenotype directory")
    parser.add_argument('--test_str_path', required=True, type=str,
                        help="Test Path to .csv file of structured model results, for phenotype directory")
    parser.add_argument('--mode', choices=['M', 'D', 'P'], required=True,
                        help="Mode: M for in-hospital mortality, D for decompensation, P for phenotyping")
    parser.add_argument('--outdir', required=True, type=str, help="Directory in which to save predictions")
    args = parser.parse_args()
    print(args)

    np.random.seed(42)

    if args.mode in ["M", "D"]:
        load_fit_save(args.mode,
                      args.str_path,
                      args.ustr_path,
                      args.test_str_path,
                      args.test_ustr_path,
                      args.outdir)
    else:
        assert(args.mode == "P")
        raise NotImplementedError




