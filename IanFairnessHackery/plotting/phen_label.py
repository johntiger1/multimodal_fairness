from tabulate import tabulate
import os
import numpy as np
import matplotlib.pyplot as plt

#Path to Dir of nice phenotype csv files
indir = "/home/administrator/00Projects/Fairness/multimodal_fairness/IanFairnessHackery/john_results/Phenotyping/train/"

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
    TASKS = list(PRED_TASKS.keys())
    TASKS.sort(key=str.lower)

    fig, ax = plt.subplots()

    xs = []
    percents = []
    tick_labels = []

    for filename in os.listdir(indir):
        prefixes = PRED_TASKS.keys()
        matches = list(filter(lambda x: filename.startswith(x), prefixes))

        # SKIP non-matches files
        if len(matches) == 0:
            continue

        # Make sure only one file for this task
        assert(not PRED_TASKS[matches[0]])
        PRED_TASKS[matches[0]] = True

        _, labels = read_file(os.path.join(indir, filename))

        percent = np.average(labels)
        percents.append(percent)
        #print(filename)
        task_index = TASKS.index(filename[:-4])
        xs.append(task_index+ 1) # Good Enough if files nice enough
        tick_labels.append(chr(task_index + ord('A')))

    plt.bar(xs, percents, tick_label=tick_labels)
    ax.set_ylabel("Percentage of Positive Cases")
    ax.set_xlabel("Task Code")
    plt.savefig("phen_label_train_percentages")

    print(tabulate({"Code": [chr(i + ord('A')) for i in range(25)], "Task": TASKS}, headers="keys", tablefmt="latex"))