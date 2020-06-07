# the phenotyping problem is annoying since you end up with 25 binary tasks; assuming that all of your prediction csv's
# are properly prefixed with the name of the task and reside within a test results folder, then this script will
# will evaluate model performance on each task, and aggregate performance

# python2 -um IanFairnessHackery.evaluate_mort_preds <PATH TO CSV FILE>

from mimic3models import metrics
import sys
import os
import numpy as np

def read_file(path):
    predictions = []
    labels = []

    with open(path, 'r') as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            vals = line.split(",")
            predictions.append(float(vals[3]))
            labels.append(int(vals[4]))

    return np.array(predictions), np.array(labels)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Must provide path to prediction csv's in id/episode/pred/label format")
        exit(-1)

    predictions, labels = read_file(os.path.join(sys.argv[1]))

    metrics.print_metrics_binary(labels, predictions)
