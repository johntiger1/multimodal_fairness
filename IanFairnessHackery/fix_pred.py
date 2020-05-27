# Quick ugly hack script to convert the baseline test output into the format we need

import os

# Set to path to predictions
READ_DECO = "../mimic3models/decompensation/test_predictions/nrk_channel_wise_lstms.n16.szc8.0.dep1.dsup.bs32.ts1.0.chunk6.test0.0810981076094.state.csv"
READ_IHM = "../mimic3models/in_hospital_mortality/test_predictions/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state.csv"
READ_PHN = "../mimic3models/phenotyping/test_predictions/nr6k_channel_wise_lstms.n16.szc8.0.d0.3.dep1.bs64.ts1.0.epoch49.test0.348234337795.state.csv"

# train predictions
READ_DECO2 = "../mimic3models/decompensation/train_predictions/nrk_channel_wise_lstms.n16.szc8.0.dep1.dsup.bs32.ts1.0.chunk6.test0.0810981076094.state.csv"
READ_IHM2 = "../mimic3models/in_hospital_mortality/train_predictions/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state.csv"
READ_PHN2 = "../mimic3models/phenotyping/train_predictions/nr6k_channel_wise_lstms.n16.szc8.0.d0.3.dep1.bs64.ts1.0.epoch49.test0.348234337795.state.csv"

# validation predictions
READ_DECO3 = "../mimic3models/decompensation/val_predictions/nrk_channel_wise_lstms.n16.szc8.0.dep1.dsup.bs32.ts1.0.chunk6.test0.0810981076094.state.csv"
READ_IHM3 = "../mimic3models/in_hospital_mortality/val_predictions/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state.csv"
READ_PHN3 = "../mimic3models/phenotyping/val_predictions/nr6k_channel_wise_lstms.n16.szc8.0.d0.3.dep1.bs64.ts1.0.epoch49.test0.348234337795.state.csv"


# Flags for which to run
CLEAN_DECO = True
CLEAN_IHM = True
CLEAN_PHN = True



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

def split_first_col (READ_PATH, header, preproc = lambda x: x):
    """
    Ugly horrible hacky code to read in the .csv at READ_PATH, and, assuming the first column is of the format
    "ID_episodeEPISODE_timeseries.csv", then splits ID and episode into 2 columns
    :p
    :param READ_PATH:
    :param preproc:
    :return:
    """
    assert(READ_PATH.endswith(".csv"))
    WRITE_PATH = READ_PATH[:-4] + "_id_ep_fmt.csv"

    with open(READ_PATH, 'r') as fr:
        with open(WRITE_PATH, 'w') as fw:
            fr.readline() #read header
            fw.write(header)

            for line in fr:
                values = line.split(",")

                # UGLY hack to do any processing
                values = preproc(values)

                name = values[0]
                assert(name.endswith("_timeseries.csv"))
                name = name[:-15]
                id_and_ep = name.split("_episode")
                assert(len(id_and_ep) == 2 and id_and_ep[0].isdigit() and id_and_ep[1].isdigit())

                del values[0]
                values = id_and_ep + values

                fw.write(",".join(values))

def decomp_proc (values):
    """
    Given the row of predictions for the decomposition mode test output, remove the pointless training zeroes from
    the time
    :param values: A row of the decomposition predictions .csv, split by commas into a list
    :return:  Same row with pointless zeros after the time removed
    """
    time = values[1]
    times = time.split(".")
    assert (len(times) == 2)
    assert (times[1] == "000000")
    values[1] = times[0]
    return values

def proc_phen(phen_path):
    n_tasks = 25

    assert (phen_path.endswith(".csv"))
    dir_path, filename = os.path.split(phen_path)
    filename = filename[:-4] + "_id_ep_fmt.csv"

    for i in range(1, n_tasks + 1):
        with open(os.path.join(dir_path, PRED_TASKS[i] + '_' + filename), 'w') as f:
            header = ["id", "episode", "prediction", "label"]
            header = ",".join(header)
            f.write(header + '\n')

    with open(phen_path, 'r') as fr:
        fr.readline()

        for inline in fr:
            inline = inline.strip()
            values = inline.split(",")
            name = values[0]
            assert (name.endswith("_timeseries.csv"))
            name = name[:-15]
            vals = name.split("_episode")
            assert (len(vals) == 2 and vals[0].isdigit() and vals[1].isdigit())

            #ts = values[1]
            predictions = values[2:2+n_tasks]
            labels = values[2+n_tasks:]
            for i in range(1, n_tasks + 1):
                with open(os.path.join(dir_path, PRED_TASKS[i] + '_' + filename), 'a') as f:
                    outline = vals.copy()
                    #line += ["{:.6f}".format(t)] # We don't need time in the output
                    outline += [predictions[i-1]]
                    outline += [labels[i-1]]
                    outline = ",".join(outline)
                    f.write(outline + '\n')

if __name__ == '__main__':
    pass

    # process decomensation
    if CLEAN_DECO:
        split_first_col(READ_DECO, "id,episode,time,prediction,label\n", decomp_proc)
        split_first_col(READ_DECO2, "id,episode,time,prediction,label\n", decomp_proc)
        split_first_col(READ_DECO3, "id,episode,time,prediction,label\n", decomp_proc)

    # process in-hospital mortality
    if CLEAN_IHM:
        split_first_col(READ_IHM, "id,episode,prediction,label\n")
        split_first_col(READ_IHM2, "id,episode,prediction,label\n")
        split_first_col(READ_IHM3, "id,episode,prediction,label\n")


    # process phenotyping
    if CLEAN_PHN:
        proc_phen(READ_PHN)
        proc_phen(READ_PHN2)
        proc_phen(READ_PHN3)




