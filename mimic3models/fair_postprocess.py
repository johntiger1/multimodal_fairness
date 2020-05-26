import numpy as np
import sys, csv
import json
from fair_classifier import *
import matplotlib.pyplot as plt

ALL_SENSITIVE = {"ETHNICITY":0, "GENDER":1, "INSURANCE":2, "RELIGION":3, "MARITIAL_STATUS":4}

def create_sensitive_dict(filename):
    sensitive_dict = {}
    csv_file = open(filename, mode='r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    for i, row in enumerate(csv_reader):
        if i == 0:
            continue
        else:
            sensitive_dict[int(row[0])] = [row[1], row[2], row[3], row[4], row[5]]

    with open('sensitive.json', 'w') as fp:
        json.dump(sensitive_dict, fp)

def load_sensitive_dict():
    with open('sensitive.json', 'r') as fp:
        data = json.load(fp)
    
    new_out = {}
    for key in data.keys():
        new_out[int(key)] = data[key]
    return new_out

def create_train_test_data(train_file, test_file, sensitive):
    train_csv = open(train_file, mode='r')
    train_csv_reader = csv.reader(train_csv, delimiter=',')
    test_csv = open(train_file, mode='r')
    test_csv_reader = csv.reader(test_csv, delimiter=',')
    sensitive_dict = load_sensitive_dict()

    def get_data(csv_reader):
        X, score, Y, sensitive_attr = [], [], [], []
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            else:
                subject_id = int(row[0])
                sensitive_attr.append(sensitive_dict[subject_id][ALL_SENSITIVE[sensitive]])
                X.append((int(row[0]), float(row[1])))
                score.append(float(row[2]))
                Y.append(int(row[3]))
        return X, score, Y, sensitive_attr

    train_X, train_score, train_Y, sens_train = get_data(train_csv_reader)
    test_X, test_score, test_Y, sens_test = get_data(test_csv_reader)

    return np.array(train_X), np.array(train_score), np.array(train_Y), np.array(sens_train, dtype='<U16'), \
            np.array(test_X), np.array(test_score), np.array(test_Y), np.array(sens_test, dtype='<U16')


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "LOAD":
        sens_file = sys.argv[2]
        create_sensitive_dict(sens_file)
        print("Loaded sensitive class info and dumped into sensitive.json")
    elif cmd == "RUN":
        train_file = sys.argv[2]
        test_file = sys.argv[3]
        sensitive_attr = sys.argv[4]
                
        assert(sensitive_attr in ALL_SENSITIVE) 
        train_X, train_score, train_Y, sens_train, \
                test_X, test_score, test_Y, sens_test = create_train_test_data(train_file, test_file, sensitive_attr)
       
        classifier = pseudo_classifier(train_X, train_Y, train_score, sens_train)
        classifier.fit(train_X, train_Y)
        classifier.get_group_confusion_matrix(sens_train, train_X, train_Y)

        my_fair_classifier = fair_classifier(train_X, train_Y, train_score, sens_train, "equalized_odds")
        my_fair_classifier.fit()
        my_fair_classifier.get_group_confusion_matrix(sens_train, train_X, train_Y)

        if len(sys.argv) == 7:
            train_dump_file = sys.argv[5]
            test_dump_file = sys.argv[6]
            
            print("Dumping post processed output into ", train_dump_file, " and ", test_dump_file)
            csv_file = open(train_dump_file, mode='w')
            csv_reader = csv.writer(csv_file, delimiter=',')
            fair_score = my_fair_classifier.predict_prob(train_X, sens_train)
            acc = 1 - (np.sum(np.power(fair_score[:,1] - train_Y, 2))/len(train_Y))

            csv_reader.writerow(["ID", "EPISODE", "PREDICTION", "LABEL"])
            for train, score, y in zip(train_X, fair_score, train_Y):
                csv_reader.writerow([str(train[0]), str(train[1]), str(score[1]), str(y)])
    
            csv_file = open(test_dump_file, mode='w')
            csv_reader = csv.writer(csv_file, delimiter=',')
            fair_score = my_fair_classifier.predict_prob(test_X, sens_test)
            csv_reader.writerow(["ID", "EPISODE", "PREDICTION (1)", "LABEL"])
            for test, score, y in zip(test_X, fair_score, test_Y):
                csv_reader.writerow([str(int(test[0])), str(test[1]), str(score[1]), str(y)])
    
    elif cmd == "PLOT":
        train_file = sys.argv[2]
        test_file = sys.argv[3]
        sensitive_attr = sys.argv[4]
        
        assert(sensitive_attr in ALL_SENSITIVE) 
        train_X, train_score, train_Y, sens_train, \
                test_X, test_score, test_Y, sens_test = create_train_test_data(train_file, test_file, sensitive_attr)
        
        base_classifier = pseudo_classifier(train_X, train_Y, train_score, sens_train)
        base_classifier.fit(train_X, train_Y)
        base_confusion = base_classifier.get_group_confusion_matrix(sens_train, train_X, train_Y)

        dp_fair_classifier = fair_classifier(train_X, train_Y, train_score, sens_train, "demographic_parity")
        dp_fair_classifier.fit()
        dp_confusion = dp_fair_classifier.get_group_confusion_matrix(sens_train, train_X, train_Y)

        eo_fair_classifier = fair_classifier(train_X, train_Y, train_score, sens_train, "equalized_odds")
        eo_fair_classifier.fit()
        eo_confusion = eo_fair_classifier.get_group_confusion_matrix(sens_train, train_X, train_Y)
        
        to_plot = ['TP Rate', 'TN Rate', 'FP Rate', 'FN Rate']
        x_axis = ["Base Classifier", "DP Classifier", "EO Classifier"]
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:gray', 'tab:olive']
        plots = []
        groups = base_confusion.keys()

        for i in range(len(to_plot)):
            fig, ax = plt.subplots()
            ax.set_xlabel("Classifier")
            ax.set_ylabel(to_plot[i])

            for j, group in enumerate(groups):
                ax.plot(x_axis, [base_confusion[group][i], dp_confusion[group][i], \
                        eo_confusion[group][i]], marker='o', linestyle='-', \
                        markersize=20, color=colors[j], label=group)
           
            plt.legend()
            plt.grid(False)
            plt.tight_layout()
            plt_name = sensitive_attr + "_" + str(to_plot[i])+".png"
            plt.savefig(plt_name)

    
