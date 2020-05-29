# the phenotyping problem is annoying since you end up with 25 binary tasks; assuming that all of your prediction csv's
# are properly prefixed with the name of the task and reside within a test results folder, then this script will
# will evaluate model performance on each task, and aggregate performance

# python2 -um IanFairnessHackery.evaluate_phenotype_preds ./IanFairnessHackery/john_results/Phenotyping/test/

from mimic3models import metrics
import sys
import os
import numpy as np

PRED_TASKS = {
    "Acute and unspecified renal failure" : False,
    "Acute cerebrovascular disease" : False,
    "Acute myocardial infarction" : False,
    "Cardiac dysrhythmias" : False,
    "Chronic kidney disease" : False,
    "Chronic obstructive pulmonary disease and bronchiectasis" : False,
    "Complications of surgical procedures or medical care" : False,
    "Conduction disorders" : False,
    "Congestive heart failure" : False,
    "nonhypertensive" : False,
    "Coronary atherosclerosis and other heart disease" : False,
    "Diabetes mellitus with complications" : False,
    "Diabetes mellitus without complication" : False,
    "Disorders of lipid metabolism" : False,
    "Essential hypertension" : False,
    "Fluid and electrolyte disorders" : False,
    "Gastrointestinal hemorrhage" : False,
    "Hypertension with complications and secondary hypertension" : False,
    "Other liver diseases" : False,
    "Other lower respiratory disease" : False,
    "Other upper respiratory disease" : False,
    "Pleurisy" : False,
    "pneumothorax" : False,
    "pulmonary collapse" : False,
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)" : False
}

def read_file(path):
    predictions = []
    labels = []

    with open(path, 'r') as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            vals = line.split(",")
            predictions.append(float(vals[2]))
            labels.append(int(vals[3]))

    return np.array(predictions), np.array(labels)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Must provide path to folder containing the prediction csv's in id/episode/pred/label format, and with" +
              " filenames that are prefixed by the condition")
        exit(-1)

    merged_pred = None
    merged_Y = None

    indir = sys.argv[1]
    for filename in os.listdir(indir):
        prefixes = PRED_TASKS.keys()
        matches = filter(lambda x: filename.startswith(x), prefixes)

        # SKIP non-matches files
        if len(matches) == 0:
            continue

        # Make sure only one file for this task
        assert(not PRED_TASKS[matches[0]])
        PRED_TASKS[matches[0]] = True

        print("Evaluating {}".format(matches[0]))

        match_pred, match_Y = read_file(os.path.join(indir, filename))

        if merged_pred is None:
            merged_pred = np.expand_dims(match_pred.copy(), axis=0)
            merged_Y = np.expand_dims(match_Y.copy(), axis=0)
        else:
            merged_pred =np.concatenate((merged_pred, np.expand_dims(match_pred, axis=0)), axis=0)
            merged_Y =np.concatenate((merged_Y, np.expand_dims(match_Y ,axis=0)), axis=0)

        #print(merged_X.shape)
        #print(merged_Y.shape)

        metrics.print_metrics_binary(match_Y, match_pred)
        print("----------------------------------------")


    print("\n==========================================")
    print("Evaluating all together:")
    metrics.print_metrics_multilabel(merged_Y.T, merged_pred.T)

    for key in PRED_TASKS:
        if PRED_TASKS[key] != True:
            print("WARNING: Data for task {} missing?".format(key))