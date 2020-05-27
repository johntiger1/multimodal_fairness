# UGLY HACKY SCRIPT FOR CONVERTING JOHN's CSV OUTPUT TO MY CSV OUTPUT

import os
import argparse
import pandas as pd

# python2 merge_and_convert.py --mode M --inpath ./john_results/final_preds_mort.csv
# python2 merge_and_convert.py --mode M --inpath ./john_results/train_final_preds_mort.csv

# python2 merge_and_convert.py --mode P --inpath ./john_results/Phenotyping/test
# python2 merge_and_convert.py --mode P --inpath ./john_results/Phenotyping/train

PRED_TASKS = {
    1: "Acute and unspecified renal failure",
    2: "Acute cerebrovascular disease",
    3: "Acute myocardial infarction",
    4: "Cardiac dysrhythmias",
    5: "Chronic kidney disease",
    6: "Chronic obstructive pulmonary disease and bronchiectasis",
    7: "Complications of surgical procedures or medical care",
    8: "Conduction disorders",
    9: "Congestive heart failure",
    10: "nonhypertensive",
    11: "Coronary atherosclerosis and other heart disease",
    12: "Diabetes mellitus with complications",
    13: "Diabetes mellitus without complication",
    14: "Disorders of lipid metabolism",
    15: "Essential hypertension",
    16: "Fluid and electrolyte disorders",
    17: "Gastrointestinal hemorrhage",
    18: "Hypertension with complications and secondary hypertension",
    19: "Other liver diseases",
    20: "Other lower respiratory disease",
    21: "Other upper respiratory disease",
    22: "Pleurisy",
    23: "pneumothorax",
    24: "pulmonary collapse",
    25: "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)"
}

NUM_TASKS = 25

def drop_cols(infile, expected_header, cols_to_keep):
    """
    Given .csv file at infile, read the file, check that the header matches expected_header, and create another file
    at the same path with _ep_id_fmt.csv suffixed which only contains the columns in cols_to_keep

    :param infile: Path to input .csv
    :param expected_header: Header that input .csv should have
    :param cols_to_keep: indices of columns to include in output
    :return: None, instead saves results as another file
    """
    dirname, out_filename = os.path.split(infile)
    assert(out_filename.endswith(".csv"))
    out_filename = out_filename[:-4] + "_id_ep_fmt.csv"
    out_path = os.path.join(dirname, out_filename)

    with open(infile, 'r') as fr:
        with open(out_path, 'w') as fw:
            header = fr.readline()
            header = header.strip()
            assert (header == expected_header)
            names = header.split(",")
            fw.write(",".join([names[col] for col in cols_to_keep]) + "\n")

            for line in fr:
                line = line.strip()
                vals = line.split(",")
                fw.write(",".join([vals[col] for col in cols_to_keep]) + '\n')

def process_phen(inpath):
    """
    Given directory inpath, read all of John's .csv's and process
    """
    first_write = True

    for filename in os.listdir(inpath):
        if not filename.startswith("predictions_"):
            continue

        with open(os.path.join(inpath, filename), 'r') as fr:
            header = fr.readline().strip()
            assert(header == ",patient_id,episode,hadm_id,time,probs_0,probs_1,probs_2,probs_3,probs_4,probs_5,probs_6,probs_7,probs_8,probs_9,probs_10,probs_11,probs_12,probs_13,probs_14,probs_15,probs_16,probs_17,probs_18,probs_19,probs_20,probs_21,probs_22,probs_23,probs_24,label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_10,label_11,label_12,label_13,label_14,label_15,label_16,label_17,label_18,label_19,label_20,label_21,label_22,label_23,label_24")

            for inline in fr:
                inline = inline.strip()
                invals = inline.split(",")

                if first_write:
                    write_mode = 'w'
                else:
                    write_mode = 'a'

                for i in range(1, NUM_TASKS + 1):
                    with open(os.path.join(inpath, PRED_TASKS[i]+".csv"), write_mode) as fw:
                        if first_write:
                            fw.write("patient_id,episode,prediction,label\n")
                        fw.write("{},{},{},{}\n".format(invals[1], invals[2], invals[4+i], invals[4+NUM_TASKS+i]))

                first_write = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['M', 'D', 'P'], required=True, help="Mode: M for in-hospital mortality, D for decompensation, P for phenotyping")
    parser.add_argument('--inpath', required=True, help="Path to .csv file in John's format; for phenotyping a folder of csv's to merge and convert")
    args = parser.parse_args()
    print(args)

    if args.mode == "M":
        drop_cols(args.inpath, ",Unnamed: 0,patient_id,episode,hadm_id,probs_0,probs_1,label,predictions", [2,3,6,7])
    elif args.mode == "P":
        process_phen(args.inpath)
    else:
        raise NotImplementedError