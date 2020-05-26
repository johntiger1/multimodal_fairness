# UGLY HACKY SCRIPT FOR CONVERTING JOHN's CSV OUTPUT TO MY CSV OUTPUT

import os
import argparse

# python merge_and_convert --mode M --inpath ./john_results/final_preds_mort.csv

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['M', 'D', 'P'], required=True, help="Mode: M for in-hospital mortality, D for decompensation, P for phenotyping")
    parser.add_argument('--inpath', required=True, help="Path to .csv file in John's format")
    args = parser.parse_args()
    print(args)

    if args.mode == "M":
        drop_cols(args.inpath, ",Unnamed: 0,patient_id,episode,hadm_id,probs_0,probs_1,label,predictions", [2,3,6,7])
    else:
        raise NotImplementedError