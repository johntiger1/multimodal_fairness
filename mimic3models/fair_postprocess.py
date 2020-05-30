import numpy as np
import sys, csv
import json
from fair_classifier import *
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    
    test_csv = open(test_file, mode='r')
    test_csv_reader = csv.reader(test_csv, delimiter=',')
    sensitive_dict = load_sensitive_dict()

    def get_data(csv_reader):
        sensitive_dict_incomplete = False
        X, score, Y, sensitive_attr = [], [], [], []
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            else:
                subject_id = int(row[0])

                if not subject_id in sensitive_dict:
                    if not sensitive_dict_incomplete:
                        sensitive_dict_incomplete = True
                        print("WARNING: Sensitive dictionary is missing patient IDs", file=sys.stderr)
                    continue

                sensitive_attr.append(sensitive_dict[subject_id][ALL_SENSITIVE[sensitive]])
                X.append((int(row[0]), float(row[1])))
                score.append(float(row[2]))
                Y.append(int(row[3]))
        return X, score, Y, sensitive_attr

    train_X, train_score, train_Y, sens_train = get_data(train_csv_reader)
    test_X, test_score, test_Y, sens_test = get_data(test_csv_reader)

    return np.array(train_X), np.array(train_score), np.array(train_Y), np.array(sens_train, dtype='<U16'), \
            np.array(test_X), np.array(test_score), np.array(test_Y), np.array(sens_test, dtype='<U16')


def get_conf_base(input):
    train_X, train_score, train_Y, sens_train, test_X, test_score, test_Y, sens_test = input
    base_classifier = pseudo_classifier(train_X, train_Y, train_score, sens_train, test_X, test_Y, test_score,
                                        sens_test)
    base_classifier.fit(train_X, train_Y)
    return base_classifier.get_group_confusion_matrix(sens_test, test_X, test_Y)

def get_conf_fair(input, fairness_type):
    train_X, train_score, train_Y, sens_train, test_X, test_score, test_Y, sens_test = input
    base_classifier = fair_classifier(train_X, train_Y, train_score, sens_train, \
                                             test_X, test_Y, test_score, sens_test, fairness_type)
    base_classifier.fit()
    return base_classifier.get_avg_group_confusion_matrix(sens_test, test_X, test_Y)

def plot_line_with_offset(ax, conf_micro_macro_pairs, groups, offset, DIAMOND_THICKNESS, to_plot, i, vert_line_style, generate_legend=False):
    for k, (model_confusion, model_micro_macro) in enumerate(conf_micro_macro_pairs):
        max_val = -1
        min_val = 10

        for group in groups:
            max_val = max(max_val, model_confusion[group][i])
            min_val = min(min_val, model_confusion[group][i])

        # If micro/macro statistics available, add diamond indicator
        if to_plot[i][9:] in model_micro_macro or to_plot[
            i] in model_micro_macro:  # TODO: Reduce Ugly hackiness
            if to_plot[i][9:] in model_micro_macro:
                micro, macro = model_micro_macro[to_plot[i][9:]]
            else:
                micro, macro = model_micro_macro[to_plot[i]]
            ax.fill((1 + k - offset,
                     1 + k - offset, 1 + k - offset - DIAMOND_THICKNESS),
                    (max_val, min_val, macro),
                    color="xkcd:light lavender")  # xkcd:light lavender

            ax.fill((1 + k - offset, 1 + k - offset + DIAMOND_THICKNESS,
                     1 + k - offset),
                    (max_val, micro, min_val),
                    color="xkcd:light grey")

            # Add outline
            ax.plot((1 + k - offset, 1 + k + - offset + DIAMOND_THICKNESS,
                     1 + k - offset),
                    (max_val, micro, min_val),
                    color="black", linewidth=DIAMOND_LINEWIDTH)

            ax.plot((1 + k - offset,
                     1 + k - offset - DIAMOND_THICKNESS, 1 + k - offset),
                    (max_val, macro, min_val), color="black", linewidth=DIAMOND_LINEWIDTH)

            # Add star to points
            ax.plot(1 + k - offset - DIAMOND_THICKNESS, macro, color="xkcd:light lavender",
                    marker='o', mec="red", markersize=DIAMOND_MARKERSIZE)
            ax.plot(1 + k - offset + DIAMOND_THICKNESS, micro, color="xkcd:light grey",
                    marker='o', mec="black", markersize=DIAMOND_MARKERSIZE)

        for j, group in enumerate(groups):
            # draw black outline
            ax.plot(1 + k - offset, model_confusion[group][i], marker=EN_MARK, \
                    markersize=MARKER_SIZE + OUTLINE_THICKNESS, color="black", mew=MEW + OUTLINE_THICKNESS)

            ax.plot(1 + k - offset, model_confusion[group][i], marker=U_MARK, \
                    markersize=MARKER_SIZE, color=colors[j], mew=MEW)

        ax.plot((1 + k - offset, 1 + k - offset), (min_val, max_val), color="black", linestyle=vert_line_style)


        if generate_legend:
            # Create Legend
            lines = [mpl.lines.Line2D([0], [0], marker=EN_MARK,
                                      markersize=MARKER_SIZE, color=colors[j], mew=MEW) for j, group in
                     enumerate(groups)]
            labels = groups.copy()

            if to_plot[i][9:] in model_micro_macro or to_plot[i] in model_micro_macro:  # TODO: Reduce Ugly hackiness
                if to_plot[i][9:] in model_micro_macro:
                    name = to_plot[i][9:]
                    micro, macro = model_micro_macro[to_plot[i][9:]]
                else:
                    name = to_plot[i]
                    micro, macro = model_micro_macro[to_plot[i]]
                lines += [mpl.lines.Line2D([0], [0], color=col,
                                           marker='o', mec=mec, markersize=DIAMOND_MARKERSIZE) for col, mec in
                          (("xkcd:light lavender", "red"), ("xkcd:light grey", "black"))]
                labels += ["Macro " + name, "Micro " + name]

            lines += [mpl.lines.Line2D([0], [0], color="black", linestyle=style) for style in ('solid', 'dashdot', 'dotted')]
            labels += ['Unstructured', 'Structured', 'Ensemble']
            plt.legend(lines, labels)

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
       
        classifier = pseudo_classifier(train_X, train_Y, train_score, sens_train, test_X, test_Y, test_score, sens_test)
        classifier.fit(train_X, train_Y)
        classifier.get_group_confusion_matrix(sens_test, test_X, test_Y)

        my_fair_classifier = fair_classifier(train_X, train_Y, train_score, sens_train, \
                test_X, test_Y, test_score, sens_test, "equalized_odds")
        my_fair_classifier.fit()
        my_fair_classifier.get_avg_group_confusion_matrix(sens_test, test_X, test_Y)

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
    
    elif cmd == "PLOT_ONE":
        train_file = sys.argv[2]
        test_file = sys.argv[3]
        sensitive_attr = sys.argv[4]
        
        assert(sensitive_attr in ALL_SENSITIVE) 
        train_X, train_score, train_Y, sens_train, \
                test_X, test_score, test_Y, sens_test = create_train_test_data(train_file, test_file, sensitive_attr)
        
        base_classifier = pseudo_classifier(train_X, train_Y, train_score, sens_train, test_X, test_Y, test_score, sens_test)
        base_classifier.fit(train_X, train_Y)
        base_confusion, _ = base_classifier.get_group_confusion_matrix(sens_test, test_X, test_Y)

        dp_fair_classifier = fair_classifier(train_X, train_Y, train_score, sens_train, \
                test_X, test_Y, test_score, sens_test, "demographic_parity")
        dp_fair_classifier.fit()
        dp_confusion = dp_fair_classifier.get_avg_group_confusion_matrix(sens_test, test_X, test_Y)

        eo_fair_classifier = fair_classifier(train_X, train_Y, train_score, sens_train, \
                test_X, test_Y, test_score, sens_test, "equalized_odds")
        eo_fair_classifier.fit()
        eo_confusion = eo_fair_classifier.get_avg_group_confusion_matrix(sens_test, test_X, test_Y)
        
        to_plot = ['TP Rate', 'TN Rate', 'FP Rate', 'FN Rate', "Accuracy"]
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
                        markersize=15, color=colors[j], label=group)
           
            plt.legend()
            plt.grid(False)
            plt.tight_layout()
            plt_name = sensitive_attr + "_" + str(to_plot[i])+".png"
            plt.savefig(plt_name)

    elif cmd == "PLOT_ALL":
        u_train_file = sys.argv[2]
        u_test_file = sys.argv[3]
        en_train_file = sys.argv[4]
        en_test_file = sys.argv[5]
        sensitive_attr = sys.argv[6]
        
        assert(sensitive_attr in ALL_SENSITIVE) 
        train_X, train_score, train_Y, sens_train, \
                test_X, test_score, test_Y, sens_test = create_train_test_data(u_train_file, u_test_file, sensitive_attr)
        en_train_X, en_train_score, en_train_Y, en_sens_train, \
                en_test_X, en_test_score, en_test_Y, en_sens_test = create_train_test_data(en_train_file, en_test_file, sensitive_attr)
       
        print("Train")
        print(train_X[0], train_score[0])
        print(en_train_X[0], en_train_score[0])
        print("Test")
        print(test_X[0], test_score[0])
        print(en_test_X[0], en_test_score[0])

        base_classifier = pseudo_classifier(train_X, train_Y, train_score, sens_train, test_X, test_Y, test_score, sens_test)
        base_classifier.fit(train_X, train_Y)
        base_confusion, base_micro_macro = base_classifier.get_group_confusion_matrix(sens_test, test_X, test_Y)
        
        en_base_classifier = pseudo_classifier(en_train_X, en_train_Y, en_train_score, en_sens_train, en_test_X, en_test_Y, en_test_score, en_sens_test)
        en_base_classifier.fit(en_train_X, en_train_Y)
        en_base_confusion, en_base_micro_macro = en_base_classifier.get_group_confusion_matrix(en_sens_test, en_test_X, en_test_Y)

        dp_fair_classifier = fair_classifier(train_X, train_Y, train_score, sens_train, \
               test_X, test_Y, test_score, sens_test, "demographic_parity")
        dp_fair_classifier.fit()
        dp_confusion, dp_micro_macro = dp_fair_classifier.get_avg_group_confusion_matrix(sens_test, test_X, test_Y)

        en_dp_fair_classifier = fair_classifier(en_train_X, en_train_Y, en_train_score, \
                en_sens_train, en_test_X, en_test_Y, en_test_score, en_sens_test, "demographic_parity")
        en_dp_fair_classifier.fit()
        en_dp_confusion, en_dp_micro_macro = en_dp_fair_classifier.get_avg_group_confusion_matrix(en_sens_test, en_test_X, en_test_Y)

        eo_fair_classifier = fair_classifier(train_X, train_Y, train_score, sens_train, \
               test_X, test_Y, test_score, sens_test, "equalized_odds")
        eo_fair_classifier.fit()
        eo_confusion, eo_micro_macro = eo_fair_classifier.get_avg_group_confusion_matrix(sens_test, test_X, test_Y)

        en_eo_fair_classifier = fair_classifier(en_train_X, en_train_Y, en_train_score, \
                en_sens_train, en_test_X, en_test_Y, en_test_score, en_sens_test, "equalized_odds")
        en_eo_fair_classifier.fit()
        en_eo_confusion, en_eo_micro_macro = en_eo_fair_classifier.get_avg_group_confusion_matrix(en_sens_test, en_test_X, en_test_Y)

        to_plot = ['Expected TP Rate', 'Expected TN Rate', 'Expected FP Rate', 'Expected FN Rate', "Expected Accuracy", "AUC"]
        x_axis = ["Base Classifier", "DP Classifier", "EO Classifier"]
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:gray', 'tab:olive']
        plots = []
        groups = base_confusion.keys()

        # Fix order
        groups = list(groups)

        for i in range(len(to_plot)):
            fig, ax = plt.subplots()
            ax.set_xlabel("Classifier")
            ax.set_ylabel(to_plot[i])

            for j, group in enumerate(groups):
                ax.plot(x_axis, [base_confusion[group][i], dp_confusion[group][i], \
                        eo_confusion[group][i]], marker='o', linestyle='-', \
                        markersize=15, color=colors[j], label=group)
            
            for j, group in enumerate(groups):
                ax.plot(x_axis, [en_base_confusion[group][i], en_dp_confusion[group][i], \
                        en_eo_confusion[group][i]], marker='x', linestyle="dotted", \
                        markersize=15, color=colors[j], label=group)
           
            plt.legend()
            plt.grid(False)
            plt.tight_layout()
            plt_name = sensitive_attr + "_" + str(to_plot[i])+".png"
            plt.savefig(plt_name)

        # Type 2 Plot
        LINE_OFFSET = 0.1
        MARKER_SIZE = 20
        U_MARK = "_"
        EN_MARK = "_"
        MEW = 4 # Marker Edge Width

        LABEL1 = True
        LABEL2 = False

        OUTLINE_THICKNESS = 2

        for i in range(len(to_plot)):
            fig, ax = plt.subplots()
            ax.set_xlabel("Classifier")
            ax.set_xticks([1,2,3])
            ax.set_xticklabels(x_axis)
            ax.set_xlim([0.5,4.5])
            ax.set_ylabel(to_plot[i])

            labelled = False
            for k, model_confusion in enumerate((base_confusion, dp_confusion, eo_confusion)):
                max_val = -1
                min_val = 10


                for j, group in enumerate(groups):
                    max_val = max(max_val, model_confusion[group][i])
                    min_val = min(min_val, model_confusion[group][i])

                    # draw black outline
                    ax.plot(1 + k - LINE_OFFSET, model_confusion[group][i], marker=EN_MARK, \
                            markersize=MARKER_SIZE + OUTLINE_THICKNESS, color="black", mew=MEW + OUTLINE_THICKNESS)

                    if LABEL1 and not labelled:
                        ax.plot(1 + k - LINE_OFFSET, model_confusion[group][i], marker=U_MARK, \
                                markersize=MARKER_SIZE, color=colors[j], label=group, mew=MEW)
                    else:
                        ax.plot(1 + k - LINE_OFFSET, model_confusion[group][i], marker=U_MARK, \
                                markersize=MARKER_SIZE, color=colors[j], mew=MEW)

                if not labelled:
                    ax.plot((1+k-LINE_OFFSET, 1+k-LINE_OFFSET),(min_val, max_val), color="black", label="Unstructured")
                else:
                    ax.plot((1 + k - LINE_OFFSET, 1 + k - LINE_OFFSET), (min_val, max_val), color="black")

                labelled = True

            labelled = False
            for k, model_confusion in enumerate((en_base_confusion, en_dp_confusion, en_eo_confusion)):
                max_val = -1
                min_val = 10

                for j, group in enumerate(groups):
                    max_val = max(max_val, model_confusion[group][i])
                    min_val = min(min_val, model_confusion[group][i])

                    # draw black outline
                    ax.plot(1 + k + LINE_OFFSET, model_confusion[group][i], marker=EN_MARK, \
                            markersize=MARKER_SIZE+OUTLINE_THICKNESS, color="black", mew=MEW+OUTLINE_THICKNESS)

                    if LABEL2 and not labelled:
                        ax.plot(1 + k + LINE_OFFSET, model_confusion[group][i], marker=EN_MARK, \
                                markersize=MARKER_SIZE, color=colors[j], label=group, mew=MEW)
                    else:
                        ax.plot(1 + k + LINE_OFFSET, model_confusion[group][i], marker=EN_MARK, \
                                markersize=MARKER_SIZE, color=colors[j], mew=MEW)

                if not labelled:
                    ax.plot((1 + k + LINE_OFFSET, 1 + k + LINE_OFFSET), (min_val, max_val), color="black",
                            label="Ensemble", linestyle="dotted")
                else:
                    ax.plot((1 + k + LINE_OFFSET, 1 + k + LINE_OFFSET), (min_val, max_val), color="black",
                            linestyle="dotted")

                labelled = True

            plt.legend()
            plt.grid(False)
            plt.tight_layout()
            plt_name = sensitive_attr + "_" + str(to_plot[i]) + "_t2.png"
            plt.savefig(plt_name)
    


        # Type 3 Plot
        LINE_OFFSET = 0.1
        MARKER_SIZE = 20
        U_MARK = "_"
        EN_MARK = "_"
        MEW = 4 # Marker Edge Width

        LABEL1 = True
        LABEL2 = False

        OUTLINE_THICKNESS = 2

        DIAMOND_THICKNESS = 0.75
        DIAMOND_LINEWIDTH = 0.5
        DIAMOND_MARKERSIZE = 2

        for i in range(len(to_plot)):
            fig, ax = plt.subplots()
            ax.set_xlabel("Classifier")
            ax.set_xticks([1,2,3])
            ax.set_xticklabels(x_axis)
            ax.set_xlim([0.5,4.5])
            ax.set_ylabel(to_plot[i])

            for k, (model_confusion, model_micro_macro) in enumerate(((base_confusion, base_micro_macro),
                                                                      (dp_confusion, dp_micro_macro),
                                                                      (eo_confusion, eo_micro_macro))):
                max_val = -1
                min_val = 10

                for group in groups:
                    max_val = max(max_val, model_confusion[group][i])
                    min_val = min(min_val, model_confusion[group][i])

                # If micro/macro statistics available, add diamond indicator
                if to_plot[i][9:] in model_micro_macro or to_plot[i] in model_micro_macro: # TODO: Reduce Ugly hackiness
                    if to_plot[i][9:] in model_micro_macro:
                        micro, macro = model_micro_macro[to_plot[i][9:]]
                    else:
                        micro, macro = model_micro_macro[to_plot[i]]
                    ax.fill((1+k-LINE_OFFSET,
                             1+k-LINE_OFFSET, 1+k - (1 + DIAMOND_THICKNESS)*LINE_OFFSET),
                            (max_val,min_val,macro),
                            color="xkcd:light lavender")

                    ax.fill((1+k-LINE_OFFSET, 1+k + (DIAMOND_THICKNESS-1) * LINE_OFFSET,
                             1+k-LINE_OFFSET),
                            (max_val, micro, min_val),
                            color="xkcd:light grey")

                    # Add outline
                    ax.plot((1 + k - LINE_OFFSET, 1 + k + (DIAMOND_THICKNESS - 1) * LINE_OFFSET,
                             1 + k - LINE_OFFSET),
                            (max_val, micro, min_val),
                            color="black", linewidth=DIAMOND_LINEWIDTH)

                    ax.plot((1 + k - LINE_OFFSET,
                             1 + k - (1 + DIAMOND_THICKNESS) * LINE_OFFSET, 1 + k - LINE_OFFSET),
                            (max_val, macro, min_val), color="black", linewidth=DIAMOND_LINEWIDTH)

                    # Add star to points
                    ax.plot(1 + k - (1 + DIAMOND_THICKNESS) * LINE_OFFSET, macro, color="xkcd:light lavender",
                            marker='o', mec="red", markersize=DIAMOND_MARKERSIZE)
                    ax.plot(1 + k + (DIAMOND_THICKNESS - 1) * LINE_OFFSET, micro, color="xkcd:light grey",
                            marker='o', mec="black", markersize=DIAMOND_MARKERSIZE)

                for j, group in enumerate(groups):
                    # draw black outline
                    ax.plot(1 + k - LINE_OFFSET, model_confusion[group][i], marker=EN_MARK, \
                            markersize=MARKER_SIZE + OUTLINE_THICKNESS, color="black", mew=MEW + OUTLINE_THICKNESS)

                    ax.plot(1 + k - LINE_OFFSET, model_confusion[group][i], marker=U_MARK, \
                            markersize=MARKER_SIZE, color=colors[j], mew=MEW)


                ax.plot((1 + k - LINE_OFFSET, 1 + k - LINE_OFFSET), (min_val, max_val), color="black")

            for k, (model_confusion, model_micro_macro) in enumerate(((en_base_confusion, en_base_micro_macro),
                                                                      (en_dp_confusion, en_dp_micro_macro),
                                                                      (en_eo_confusion, en_eo_micro_macro))):
                max_val = -1
                min_val = 10

                for group in groups:
                    max_val = max(max_val, model_confusion[group][i])
                    min_val = min(min_val, model_confusion[group][i])

                # If micro/macro statistics available, add diamond indicator
                if to_plot[i][9:] in model_micro_macro or to_plot[i] in model_micro_macro:  # TODO: Reduce Ugly hackiness
                    if to_plot[i][9:] in model_micro_macro:
                        micro, macro = model_micro_macro[to_plot[i][9:]]
                    else:
                        micro, macro = model_micro_macro[to_plot[i]]
                    ax.fill((1+k+LINE_OFFSET,
                             1+k+LINE_OFFSET, 1+k - (-1 + DIAMOND_THICKNESS)*LINE_OFFSET),
                            (max_val, min_val, macro),
                            color="xkcd:light lavender")

                    ax.fill((1+k+LINE_OFFSET, 1+k + (DIAMOND_THICKNESS+1) * LINE_OFFSET,
                             1+k+LINE_OFFSET),
                            (max_val, micro, min_val),
                            color="xkcd:light grey")

                    # Add outline
                    ax.plot((1 + k + LINE_OFFSET, 1 + k + (DIAMOND_THICKNESS + 1) * LINE_OFFSET,
                             1 + k + LINE_OFFSET),
                            (max_val, micro, min_val),
                            color="black", linewidth=DIAMOND_LINEWIDTH)

                    ax.plot((1 + k + LINE_OFFSET,
                             1 + k - (-1 + DIAMOND_THICKNESS) * LINE_OFFSET, 1 + k + LINE_OFFSET),
                            (max_val, macro, min_val), color="black", linewidth=DIAMOND_LINEWIDTH)

                    # Add star to points
                    ax.plot(1 + k - (-1 + DIAMOND_THICKNESS) * LINE_OFFSET, macro, color="xkcd:light lavender",
                            marker='o', mec="red", markersize=DIAMOND_MARKERSIZE)
                    ax.plot(1 + k + (DIAMOND_THICKNESS + 1) * LINE_OFFSET, micro, color="xkcd:light grey", marker='o',
                            mec="black", markersize=DIAMOND_MARKERSIZE)

                for j, group in enumerate(groups):
                    # draw black outline
                    ax.plot(1 + k + LINE_OFFSET, model_confusion[group][i], marker=EN_MARK, \
                            markersize=MARKER_SIZE+OUTLINE_THICKNESS, color="black", mew=MEW+OUTLINE_THICKNESS)

                    ax.plot(1 + k + LINE_OFFSET, model_confusion[group][i], marker=EN_MARK, \
                            markersize=MARKER_SIZE, color=colors[j], mew=MEW)


                ax.plot((1 + k + LINE_OFFSET, 1 + k + LINE_OFFSET), (min_val, max_val), color="black",
                        linestyle="dotted")



            # Create Legend
            lines = [mpl.lines.Line2D([0], [0], marker=EN_MARK,
                                             markersize=MARKER_SIZE, color=colors[j], mew=MEW) for j, group in enumerate(groups)]
            labels = groups.copy()

            if to_plot[i][9:] in model_micro_macro or to_plot[i] in model_micro_macro:  # TODO: Reduce Ugly hackiness
                if to_plot[i][9:] in model_micro_macro:
                    name = to_plot[i][9:]
                    micro, macro = model_micro_macro[to_plot[i][9:]]
                else:
                    name = to_plot[i]
                    micro, macro = model_micro_macro[to_plot[i]]
                lines += [mpl.lines.Line2D([0],[0],color=col,
                                marker='o', mec=mec, markersize=DIAMOND_MARKERSIZE) for col, mec in (("xkcd:light lavender", "red"),("xkcd:light grey", "black"))]
                labels += ["Macro "+name, "Micro "+name]

            lines += [mpl.lines.Line2D([0], [0], color="black", linestyle=style) for style in ('solid', 'dotted')]
            labels += ['Unstructured', 'Ensemble']
            plt.legend(lines, labels)

            plt.grid(False)
            plt.tight_layout()
            plt_name = sensitive_attr + "_" + str(to_plot[i]) + "_t3.png"
            plt.savefig(plt_name)


    elif cmd == "PLOT_ALL2":
        u_train_file = sys.argv[2]
        u_test_file = sys.argv[3]
        s_train_file = sys.argv[4]
        s_test_file = sys.argv[5]
        en_train_file = sys.argv[6]
        en_test_file = sys.argv[7]
        sensitive_attr = sys.argv[8]

        assert (sensitive_attr in ALL_SENSITIVE)

        train_X, train_score, train_Y, sens_train, \
        test_X, test_score, test_Y, sens_test = create_train_test_data(u_train_file, u_test_file, sensitive_attr)
        en_train_X, en_train_score, en_train_Y, en_sens_train, \
        en_test_X, en_test_score, en_test_Y, en_sens_test = create_train_test_data(en_train_file, en_test_file, sensitive_attr)

        print("Train")
        print(train_X[0], train_score[0])
        print(en_train_X[0], en_train_score[0])
        print("Test")
        print(test_X[0], test_score[0])
        print(en_test_X[0], en_test_score[0])


        # Get Base classifier performance
        base_confusion, base_micro_macro = get_conf_base(create_train_test_data(u_train_file, u_test_file, sensitive_attr))
        en_base_confusion, en_base_micro_macro = get_conf_base(create_train_test_data(en_train_file, en_test_file, sensitive_attr))
        s_base_confusion, s_base_micro_macro = get_conf_base(create_train_test_data(s_train_file, s_test_file, sensitive_attr))

        dp_confusion, dp_micro_macro = get_conf_fair(create_train_test_data(u_train_file, u_test_file, sensitive_attr), "demographic_parity")
        s_dp_confusion, s_dp_micro_macro = get_conf_fair(create_train_test_data(s_train_file, s_test_file, sensitive_attr), "demographic_parity")
        en_dp_confusion, en_dp_micro_macro = get_conf_fair(create_train_test_data(en_train_file, en_test_file, sensitive_attr), "demographic_parity")

        eo_confusion, eo_micro_macro = get_conf_fair(create_train_test_data(u_train_file, u_test_file, sensitive_attr), "equalized_odds")
        s_eo_confusion, s_eo_micro_macro = get_conf_fair(create_train_test_data(s_train_file, s_test_file, sensitive_attr), "equalized_odds")
        en_eo_confusion, en_eo_micro_macro = get_conf_fair(create_train_test_data(en_train_file, en_test_file, sensitive_attr), "equalized_odds")

        to_plot = ['Expected TP Rate', 'Expected TN Rate', 'Expected FP Rate', 'Expected FN Rate', "Expected Accuracy",
                   "AUC"]
        x_axis = ["Base Classifier", "DP Classifier", "EO Classifier"]
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:gray', 'tab:olive']
        plots = []
        groups = base_confusion.keys()

        # Fix order
        groups = list(groups)

        # Type 3 Plot
        LINE_OFFSET = 0.2
        MARKER_SIZE = 20
        U_MARK = "_"
        EN_MARK = "_"
        MEW = 4  # Marker Edge Width

        LABEL1 = True
        LABEL2 = False

        OUTLINE_THICKNESS = 2

        DIAMOND_THICKNESS = 0.3 * LINE_OFFSET
        DIAMOND_LINEWIDTH = 0.5
        DIAMOND_MARKERSIZE = 2

        for i in range(len(to_plot)):
            fig, ax = plt.subplots()
            ax.set_xlabel("Classifier")
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(x_axis)
            ax.set_xlim([0.5, 4.5])
            ax.set_ylabel(to_plot[i])

            # ((base_confusion, base_micro_macro), (dp_confusion, dp_micro_macro), (eo_confusion, eo_micro_macro))
            # LINE_OFFSET

            plot_line_with_offset(ax,
                                  ((base_confusion, base_micro_macro),
                                       (dp_confusion, dp_micro_macro),
                                       (eo_confusion, eo_micro_macro)),
                                  groups,
                                  LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'solid')

            plot_line_with_offset(ax,
                                  ((s_base_confusion, s_base_micro_macro),
                                   (s_dp_confusion, s_dp_micro_macro),
                                   (s_eo_confusion, s_eo_micro_macro)),
                                  groups,
                                  0,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dashdot')

            plot_line_with_offset(ax,
                                  ((en_base_confusion, en_base_micro_macro),
                                   (en_dp_confusion, en_dp_micro_macro),
                                   (en_eo_confusion, en_eo_micro_macro)),
                                  groups,
                                  -1*LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dotted',
                                  generate_legend=True)

            plt.grid(False)
            plt.tight_layout()
            plt_name = sensitive_attr + "_" + str(to_plot[i]) + "_t4.png"
            plt.savefig(plt_name)

    elif cmd == "PLOT_ALL3":
        # unstructured
        u_train_file = sys.argv[2]
        u_test_file = sys.argv[3]

        # structured
        s_train_file = sys.argv[4]
        s_test_file = sys.argv[5]

        # ensemble
        en_train_file = sys.argv[6]
        en_test_file = sys.argv[7]

        # Debiased Word Embeds
        d_train_file = sys.argv[8]
        d_test_file = sys.argv[9]

        # Debiased Word Embeds Ensemble
        den_train_file = sys.argv[10]
        den_test_file = sys.argv[11]

        sensitive_attr = sys.argv[12]

        assert (sensitive_attr in ALL_SENSITIVE)


        # Get Base classifier performance
        base_confusion, base_micro_macro = get_conf_base(create_train_test_data(u_train_file, u_test_file, sensitive_attr))
        en_base_confusion, en_base_micro_macro = get_conf_base(create_train_test_data(en_train_file, en_test_file, sensitive_attr))
        s_base_confusion, s_base_micro_macro = get_conf_base(create_train_test_data(s_train_file, s_test_file, sensitive_attr))
        d_base_confusion, d_base_micro_macro = get_conf_base(create_train_test_data(d_train_file, d_test_file, sensitive_attr))
        den_base_confusion, den_base_micro_macro = get_conf_base(create_train_test_data(den_train_file, den_test_file, sensitive_attr))

        dp_confusion, dp_micro_macro = get_conf_fair(create_train_test_data(u_train_file, u_test_file, sensitive_attr), "demographic_parity")
        s_dp_confusion, s_dp_micro_macro = get_conf_fair(create_train_test_data(s_train_file, s_test_file, sensitive_attr), "demographic_parity")
        en_dp_confusion, en_dp_micro_macro = get_conf_fair(create_train_test_data(en_train_file, en_test_file, sensitive_attr), "demographic_parity")

        eo_confusion, eo_micro_macro = get_conf_fair(create_train_test_data(u_train_file, u_test_file, sensitive_attr), "equalized_odds")
        s_eo_confusion, s_eo_micro_macro = get_conf_fair(create_train_test_data(s_train_file, s_test_file, sensitive_attr), "equalized_odds")
        en_eo_confusion, en_eo_micro_macro = get_conf_fair(create_train_test_data(en_train_file, en_test_file, sensitive_attr), "equalized_odds")

        to_plot = ['Expected TP Rate', 'Expected TN Rate', 'Expected FP Rate', 'Expected FN Rate', "Expected Accuracy",
                   "AUC"]
        x_axis = ["Base Classifier", "DP Classifier", "EO Classifier"]
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:gray', 'tab:olive']
        plots = []
        groups = base_confusion.keys()

        # Fix order
        groups = list(groups)

        # Type 3 Plot
        LINE_OFFSET = 0.2
        MARKER_SIZE = 20
        U_MARK = "_"
        EN_MARK = "_"
        MEW = 4  # Marker Edge Width

        LABEL1 = True
        LABEL2 = False

        OUTLINE_THICKNESS = 2

        DIAMOND_THICKNESS = 0.3 * LINE_OFFSET
        DIAMOND_LINEWIDTH = 0.5
        DIAMOND_MARKERSIZE = 2

        for i in range(len(to_plot)):
            fig, ax = plt.subplots()
            ax.set_xlabel("Classifier")
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(x_axis + ["Debiased WE"])
            #ax.set_xlim([0.5, 6])
            ax.set_ylabel(to_plot[i])

            plot_line_with_offset(ax,
                                  ((base_confusion, base_micro_macro),
                                       (dp_confusion, dp_micro_macro),
                                       (eo_confusion, eo_micro_macro)),
                                  groups,
                                  LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'solid')

            plot_line_with_offset(ax,
                                  ((s_base_confusion, s_base_micro_macro),
                                   (s_dp_confusion, s_dp_micro_macro),
                                   (s_eo_confusion, s_eo_micro_macro)),
                                  groups,
                                  0,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dashdot')

            plot_line_with_offset(ax,
                                  ((en_base_confusion, en_base_micro_macro),
                                   (en_dp_confusion, en_dp_micro_macro),
                                   (en_eo_confusion, en_eo_micro_macro)),
                                  groups,
                                  -1*LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dotted')

            ax.axvline(x=3.5, color="black")

            plot_line_with_offset(ax,
                                  ((d_base_confusion, d_base_micro_macro),),
                                  groups,
                                  -3 + LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'solid')

            plot_line_with_offset(ax,
                                  ((den_base_confusion, den_base_micro_macro),),
                                  groups,
                                  -3 + -1 * LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dotted', generate_legend=True)

            plt.grid(False)
            plt.tight_layout()
            plt_name = sensitive_attr + "_" + str(to_plot[i]) + "_t4.png"
            plt.savefig(plt_name)

    elif cmd == "PLOT_ALL4":
        # unstructured
        u_train_file = sys.argv[2]
        u_test_file = sys.argv[3]

        # structured
        s_train_file = sys.argv[4]
        s_test_file = sys.argv[5]

        # ensemble
        en_train_file = sys.argv[6]
        en_test_file = sys.argv[7]

        # Debiased Word Embeds
        d_train_file = sys.argv[8]
        d_test_file = sys.argv[9]

        # Debiased Word Embeds Ensemble
        den_train_file = sys.argv[10]
        den_test_file = sys.argv[11]

        sensitive_attr = sys.argv[12]

        assert (sensitive_attr in ALL_SENSITIVE)


        # Get Base classifier performance
        base_confusion, base_micro_macro = get_conf_base(create_train_test_data(u_train_file, u_test_file, sensitive_attr))
        en_base_confusion, en_base_micro_macro = get_conf_base(create_train_test_data(en_train_file, en_test_file, sensitive_attr))
        s_base_confusion, s_base_micro_macro = get_conf_base(create_train_test_data(s_train_file, s_test_file, sensitive_attr))
        d_base_confusion, d_base_micro_macro = get_conf_base(create_train_test_data(d_train_file, d_test_file, sensitive_attr))
        den_base_confusion, den_base_micro_macro = get_conf_base(create_train_test_data(den_train_file, den_test_file, sensitive_attr))

        dp_confusion, dp_micro_macro = get_conf_fair(create_train_test_data(u_train_file, u_test_file, sensitive_attr), "demographic_parity")
        s_dp_confusion, s_dp_micro_macro = get_conf_fair(create_train_test_data(s_train_file, s_test_file, sensitive_attr), "demographic_parity")
        en_dp_confusion, en_dp_micro_macro = get_conf_fair(create_train_test_data(en_train_file, en_test_file, sensitive_attr), "demographic_parity")

        eo_confusion, eo_micro_macro = get_conf_fair(create_train_test_data(u_train_file, u_test_file, sensitive_attr), "equalized_odds")
        s_eo_confusion, s_eo_micro_macro = get_conf_fair(create_train_test_data(s_train_file, s_test_file, sensitive_attr), "equalized_odds")
        en_eo_confusion, en_eo_micro_macro = get_conf_fair(create_train_test_data(en_train_file, en_test_file, sensitive_attr), "equalized_odds")

        to_plot = ['Expected TP Rate', 'Expected TN Rate', 'Expected FP Rate', 'Expected FN Rate', "Expected Accuracy",
                   "AUC"]
        x_axis = ["Base Classifier", "DP Classifier", "EO Classifier"]
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:gray', 'tab:olive']
        plots = []
        groups = base_confusion.keys()

        # Fix order
        groups = list(groups)

        # Type 3 Plot
        LINE_OFFSET = 0.2
        MARKER_SIZE = 20
        U_MARK = "_"
        EN_MARK = "_"
        MEW = 4  # Marker Edge Width

        LABEL1 = True
        LABEL2 = False

        OUTLINE_THICKNESS = 2

        DIAMOND_THICKNESS = 0.3 * LINE_OFFSET
        DIAMOND_LINEWIDTH = 0.5
        DIAMOND_MARKERSIZE = 2

        fig, axs = plt.subplots(1,3)

        for i in range(len(to_plot)):

            if to_plot[i] == 'Expected TP Rate':
                ax = axs[0]
            elif to_plot[i] == 'Expected TN Rate':
                ax = axs[1]
            elif to_plot[i] == "AUC":
                ax = axs[2]
            else:
                continue


            ax.set_xlabel("Classifier")
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(x_axis + ["Debiased WE"])
            #ax.set_xlim([0.5, 6])
            ax.set_ylabel(to_plot[i])

            plot_line_with_offset(ax,
                                  ((base_confusion, base_micro_macro),
                                       (dp_confusion, dp_micro_macro),
                                       (eo_confusion, eo_micro_macro)),
                                  groups,
                                  LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'solid')

            plot_line_with_offset(ax,
                                  ((s_base_confusion, s_base_micro_macro),
                                   (s_dp_confusion, s_dp_micro_macro),
                                   (s_eo_confusion, s_eo_micro_macro)),
                                  groups,
                                  0,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dashdot')

            plot_line_with_offset(ax,
                                  ((en_base_confusion, en_base_micro_macro),
                                   (en_dp_confusion, en_dp_micro_macro),
                                   (en_eo_confusion, en_eo_micro_macro)),
                                  groups,
                                  -1*LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dotted')

            ax.axvline(x=3.5, color="black")

            plot_line_with_offset(ax,
                                  ((d_base_confusion, d_base_micro_macro),),
                                  groups,
                                  -3 + LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'solid')

            plot_line_with_offset(ax,
                                  ((den_base_confusion, den_base_micro_macro),),
                                  groups,
                                  -3 + -1 * LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dotted',
                                  generate_legend=(to_plot[i] == 'AUC'))

        plt.grid(False)
        plt.tight_layout()
        plt_name = sensitive_attr + "_t5.png"
        fig.set_size_inches(24, 8)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=None)
        plt.savefig(plt_name)

    elif cmd == "PLOT_ALL5": # TN,TP and AUC together, with legend, for u,s,en,d,den; NO DP!
        # unstructured
        u_train_file = sys.argv[2]
        u_test_file = sys.argv[3]

        # structured
        s_train_file = sys.argv[4]
        s_test_file = sys.argv[5]

        # ensemble
        en_train_file = sys.argv[6]
        en_test_file = sys.argv[7]

        # Debiased Word Embeds
        d_train_file = sys.argv[8]
        d_test_file = sys.argv[9]

        # Debiased Word Embeds Ensemble
        den_train_file = sys.argv[10]
        den_test_file = sys.argv[11]

        sensitive_attr = sys.argv[12]

        assert (sensitive_attr in ALL_SENSITIVE)


        # Get Base classifier performance
        base_confusion, base_micro_macro = get_conf_base(create_train_test_data(u_train_file, u_test_file, sensitive_attr))
        en_base_confusion, en_base_micro_macro = get_conf_base(create_train_test_data(en_train_file, en_test_file, sensitive_attr))
        s_base_confusion, s_base_micro_macro = get_conf_base(create_train_test_data(s_train_file, s_test_file, sensitive_attr))
        d_base_confusion, d_base_micro_macro = get_conf_base(create_train_test_data(d_train_file, d_test_file, sensitive_attr))
        den_base_confusion, den_base_micro_macro = get_conf_base(create_train_test_data(den_train_file, den_test_file, sensitive_attr))

        # EO confusion
        eo_confusion, eo_micro_macro = get_conf_fair(create_train_test_data(u_train_file, u_test_file, sensitive_attr), "equalized_odds")
        s_eo_confusion, s_eo_micro_macro = get_conf_fair(create_train_test_data(s_train_file, s_test_file, sensitive_attr), "equalized_odds")
        en_eo_confusion, en_eo_micro_macro = get_conf_fair(create_train_test_data(en_train_file, en_test_file, sensitive_attr), "equalized_odds")

        to_plot = ['Expected TP Rate', 'Expected TN Rate', 'Expected FP Rate', 'Expected FN Rate', "Expected Accuracy",
                   "AUC"]
        x_axis = ["Base Classifier", "EO Classifier"]
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:gray', 'tab:olive']
        plots = []
        groups = base_confusion.keys()

        # Fix order
        groups = list(groups)

        # Type 3 Plot
        LINE_OFFSET = 0.2
        MARKER_SIZE = 20
        U_MARK = "_"
        EN_MARK = "_"
        MEW = 4  # Marker Edge Width

        LABEL1 = True
        LABEL2 = False

        OUTLINE_THICKNESS = 2

        DIAMOND_THICKNESS = 0.3 * LINE_OFFSET
        DIAMOND_LINEWIDTH = 0.5
        DIAMOND_MARKERSIZE = 2

        fig, axs = plt.subplots(1,3)

        for i in range(len(to_plot)):

            if to_plot[i] == 'Expected TP Rate':
                ax = axs[0]
            elif to_plot[i] == 'Expected TN Rate':
                ax = axs[1]
            elif to_plot[i] == "AUC":
                ax = axs[2]
            else:
                continue


            ax.set_xlabel("Classifier")
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(x_axis + ["Debiased WE"])
            ax.set_xlim([0.5, 3.5])
            ax.set_ylabel(to_plot[i])

            plot_line_with_offset(ax,
                                  ((base_confusion, base_micro_macro),
                                       (eo_confusion, eo_micro_macro)),
                                  groups,
                                  LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'solid')

            plot_line_with_offset(ax,
                                  ((s_base_confusion, s_base_micro_macro),
                                   (s_eo_confusion, s_eo_micro_macro)),
                                  groups,
                                  0,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dashdot')

            plot_line_with_offset(ax,
                                  ((en_base_confusion, en_base_micro_macro),
                                   (en_eo_confusion, en_eo_micro_macro)),
                                  groups,
                                  -1*LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dotted')

            ax.axvline(x=2.5, color="black", lw=2)
            ax.axvline(x=2.5, color="white", lw=0.5)

            plot_line_with_offset(ax,
                                  ((d_base_confusion, d_base_micro_macro),),
                                  groups,
                                  -2 + LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'solid')

            plot_line_with_offset(ax,
                                  ((den_base_confusion, den_base_micro_macro),),
                                  groups,
                                  -2 + -1 * LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dotted')

        lines = [mpl.lines.Line2D([0], [0], marker=EN_MARK,
                                  markersize=MARKER_SIZE, color=colors[j], mew=MEW) for j, group in
                 enumerate(groups)]
        labels = groups.copy()

        lines += [mpl.lines.Line2D([0], [0], color=col,
                                   marker='o', mec=mec, markersize=DIAMOND_MARKERSIZE) for col, mec in
                  (("xkcd:light lavender", "red"), ("xkcd:light grey", "black"))]
        labels += ["Macro ", "Micro "]

        lines += [mpl.lines.Line2D([0], [0], color="black", linestyle=style) for style in
                  ('solid', 'dashdot', 'dotted')]
        labels += ['Unstructured', 'Structured', 'Ensemble']

        plt.grid(False)
        plt.tight_layout()
        plt_name = sensitive_attr + "_t6.png"


        if len(lines) <= 9:
            fig.legend(lines, labels, ncol=len(lines), loc='lower center')
            fig.set_size_inches(12, 4.4)
            plt.subplots_adjust(left=0.07, bottom=0.2, right=0.98, top=None, wspace=0.2, hspace=None)
        else:
            fig.legend(lines, labels, ncol=9, loc='lower center')
            fig.set_size_inches(12, 4.6)
            plt.subplots_adjust(left=0.07, bottom=0.25, right=0.98, top=None, wspace=0.2, hspace=None)
        plt.savefig(plt_name)

    elif cmd == "PLOT_HACK": # Modify PLOT_ALL5 to remove the unstructured model

        # structured
        s_train_file = sys.argv[2]
        s_test_file = sys.argv[3]

        # ensemble
        en_train_file = sys.argv[4]
        en_test_file = sys.argv[5]

        # Debiased Word Embeds
        d_train_file = sys.argv[6]
        d_test_file = sys.argv[7]

        # Debiased Word Embeds Ensemble
        den_train_file = sys.argv[8]
        den_test_file = sys.argv[9]

        sensitive_attr = sys.argv[10]

        assert (sensitive_attr in ALL_SENSITIVE)


        # Get Base classifier performance
        en_base_confusion, en_base_micro_macro = get_conf_base(create_train_test_data(en_train_file, en_test_file, sensitive_attr))
        s_base_confusion, s_base_micro_macro = get_conf_base(create_train_test_data(s_train_file, s_test_file, sensitive_attr))
        d_base_confusion, d_base_micro_macro = get_conf_base(create_train_test_data(d_train_file, d_test_file, sensitive_attr))
        den_base_confusion, den_base_micro_macro = get_conf_base(create_train_test_data(den_train_file, den_test_file, sensitive_attr))

        # EO confusion
        s_eo_confusion, s_eo_micro_macro = get_conf_fair(create_train_test_data(s_train_file, s_test_file, sensitive_attr), "equalized_odds")
        en_eo_confusion, en_eo_micro_macro = get_conf_fair(create_train_test_data(en_train_file, en_test_file, sensitive_attr), "equalized_odds")

        to_plot = ['Expected TP Rate', 'Expected TN Rate', 'Expected FP Rate', 'Expected FN Rate', "Expected Accuracy",
                   "AUC"]
        x_axis = ["Base Classifier", "EO Classifier"]
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:gray', 'tab:olive']
        plots = []
        groups = s_base_confusion.keys()

        # Fix order
        groups = list(groups)

        # Type 3 Plot
        LINE_OFFSET = 0.2
        MARKER_SIZE = 20
        U_MARK = "_"
        EN_MARK = "_"
        MEW = 4  # Marker Edge Width

        LABEL1 = True
        LABEL2 = False

        OUTLINE_THICKNESS = 2

        DIAMOND_THICKNESS = 0.3 * LINE_OFFSET
        DIAMOND_LINEWIDTH = 0.5
        DIAMOND_MARKERSIZE = 2

        fig, axs = plt.subplots(1,3)

        for i in range(len(to_plot)):

            if to_plot[i] == 'Expected TP Rate':
                ax = axs[0]
            elif to_plot[i] == 'Expected TN Rate':
                ax = axs[1]
            elif to_plot[i] == "AUC":
                ax = axs[2]
            else:
                continue


            ax.set_xlabel("Classifier")
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(x_axis + ["Debiased WE"])
            ax.set_xlim([0.5, 3.5])
            ax.set_ylabel(to_plot[i])

            plot_line_with_offset(ax,
                                  ((s_base_confusion, s_base_micro_macro),
                                   (s_eo_confusion, s_eo_micro_macro)),
                                  groups,
                                  LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dashdot')

            plot_line_with_offset(ax,
                                  ((en_base_confusion, en_base_micro_macro),
                                   (en_eo_confusion, en_eo_micro_macro)),
                                  groups,
                                  -1*LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dotted')

            ax.axvline(x=2.5, color="black", lw=2)
            ax.axvline(x=2.5, color="white", lw=0.5)

            plot_line_with_offset(ax,
                                  ((d_base_confusion, d_base_micro_macro),),
                                  groups,
                                  -2 + LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'solid')

            plot_line_with_offset(ax,
                                  ((den_base_confusion, den_base_micro_macro),),
                                  groups,
                                  -2 + -1 * LINE_OFFSET,
                                  DIAMOND_THICKNESS,
                                  to_plot,
                                  i,
                                  'dotted')

        lines = [mpl.lines.Line2D([0], [0], marker=EN_MARK,
                                  markersize=MARKER_SIZE, color=colors[j], mew=MEW) for j, group in
                 enumerate(groups)]
        labels = groups.copy()

        lines += [mpl.lines.Line2D([0], [0], color=col,
                                   marker='o', mec=mec, markersize=DIAMOND_MARKERSIZE) for col, mec in
                  (("xkcd:light lavender", "red"), ("xkcd:light grey", "black"))]
        labels += ["Macro ", "Micro "]

        lines += [mpl.lines.Line2D([0], [0], color="black", linestyle=style) for style in
                  ('solid', 'dashdot', 'dotted')]
        labels += ['Unstructured', 'Structured', 'Ensemble']

        plt.grid(False)
        plt.tight_layout()
        plt_name = sensitive_attr + "_hack.png"


        if len(lines) <= 9:
            fig.legend(lines, labels, ncol=len(lines), loc='lower center')
            fig.set_size_inches(12, 4.4)
            plt.subplots_adjust(left=0.07, bottom=0.2, right=0.98, top=None, wspace=0.2, hspace=None)
        else:
            fig.legend(lines, labels, ncol=9, loc='lower center')
            fig.set_size_inches(12, 4.6)
            plt.subplots_adjust(left=0.07, bottom=0.25, right=0.98, top=None, wspace=0.2, hspace=None)
        plt.savefig(plt_name)

    elif cmd == "GENERATE_LEGEND":
        train_file = sys.argv[2]
        test_file = sys.argv[3]
        sensitive_attr = sys.argv[4]

        assert (sensitive_attr in ALL_SENSITIVE)
        train_X, train_score, train_Y, sens_train, \
        test_X, test_score, test_Y, sens_test = create_train_test_data(train_file, test_file, sensitive_attr)

        base_classifier = pseudo_classifier(train_X, train_Y, train_score, sens_train, test_X, test_Y, test_score,
                                            sens_test)
        base_classifier.fit(train_X, train_Y)
        base_confusion, _ = base_classifier.get_group_confusion_matrix(sens_test, test_X, test_Y)

        groups = list(base_confusion.keys())
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:gray', 'tab:olive']

        fig, ax = plt.subplots()

        # Type 3 Plot
        LINE_OFFSET = 0.2
        MARKER_SIZE = 20
        U_MARK = "_"
        EN_MARK = "_"
        MEW = 4  # Marker Edge Width

        LABEL1 = True
        LABEL2 = False

        OUTLINE_THICKNESS = 2

        DIAMOND_THICKNESS = 0.3 * LINE_OFFSET
        DIAMOND_LINEWIDTH = 0.5
        DIAMOND_MARKERSIZE = 2

        # Create Legend
        lines = [mpl.lines.Line2D([0], [0], marker=EN_MARK,
                                  markersize=MARKER_SIZE, color=colors[j], mew=MEW) for j, group in
                 enumerate(groups)]
        labels = groups.copy()

        lines += [mpl.lines.Line2D([0], [0], color="black", linestyle=style) for style in
                  ('solid', 'dashdot', 'dotted')]
        labels += ['Unstructured', 'Structured', 'Ensemble']
        plt.legend(lines, labels)
        plt.grid(False)
        plt.tight_layout()
        plt_name = sensitive_attr + "_legend1.png"
        plt.savefig(plt_name)


        fig, ax = plt.subplots()
        # Create Legend
        lines = [mpl.lines.Line2D([0], [0], marker=EN_MARK,
                                  markersize=MARKER_SIZE, color=colors[j], mew=MEW) for j, group in
                 enumerate(groups)]
        labels = groups.copy()

        lines += [mpl.lines.Line2D([0], [0], color=col,
                                   marker='o', mec=mec, markersize=DIAMOND_MARKERSIZE) for col, mec in
                  (("xkcd:light lavender", "red"), ("xkcd:light grey", "black"))]
        labels += ["Macro ", "Micro "]

        lines += [mpl.lines.Line2D([0], [0], color="black", linestyle=style) for style in
                  ('solid', 'dashdot', 'dotted')]
        labels += ['Unstructured', 'Structured', 'Ensemble']
        plt.legend(lines, labels)
        plt.grid(False)
        plt.tight_layout()
        plt_name = sensitive_attr + "_legend2.png"
        plt.savefig(plt_name)


        fig, ax = plt.subplots()
        # Create Legend
        lines = [mpl.lines.Line2D([0], [0], marker=EN_MARK,
                                  markersize=MARKER_SIZE, color=colors[j], mew=MEW) for j, group in
                 enumerate(groups)]
        labels = groups.copy()

        lines += [mpl.lines.Line2D([0], [0], color="black", linestyle=style) for style in
                  ('solid', 'dashdot', 'dotted')]
        labels += ['Unstructured', 'Structured', 'Ensemble']
        leg=plt.legend(lines, labels, ncol=len(lines))
        plt.grid(False)
        plt_name = sensitive_attr + "_legend3.png"
        fig.set_size_inches(20, 1)
        plt.savefig(plt_name)


        fig, ax = plt.subplots()
        # Create Legend
        lines = [mpl.lines.Line2D([0], [0], marker=EN_MARK,
                                  markersize=MARKER_SIZE, color=colors[j], mew=MEW) for j, group in
                 enumerate(groups)]
        labels = groups.copy()

        lines += [mpl.lines.Line2D([0], [0], color=col,
                                   marker='o', mec=mec, markersize=DIAMOND_MARKERSIZE) for col, mec in
                  (("xkcd:light lavender", "red"), ("xkcd:light grey", "black"))]
        labels += ["Macro ", "Micro "]

        lines += [mpl.lines.Line2D([0], [0], color="black", linestyle=style) for style in
                  ('solid', 'dashdot', 'dotted')]
        labels += ['Unstructured', 'Structured', 'Ensemble']
        plt.legend(lines, labels, ncol=len(lines))
        plt.grid(False)
        plt_name = sensitive_attr + "_legend4.png"
        fig.set_size_inches(20, 1)
        plt.savefig(plt_name)