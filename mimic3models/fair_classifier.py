import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import roc_auc_score

class pseudo_classifier:
    """ Note that the actual classifier is already trained (unaware as it ought to be) and 
    post-processing does not require access to the underlying features. So this class is 
    basically a wrapper around that prediction so we can pass it nicely onto the 
    fairlearn framework
    """
    def __init__(self, train_X, train_Y, train_score_Y, sensitive_train, \
            test_X=None, test_Y=None, test_score_Y=None, sensitive_test=None, \
            sensitive_features_dict=None):
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_score_Y = train_score_Y
        self.sensitive_train = sensitive_train
        
        self.test_X = test_X
        self.test_Y = test_Y
        self.test_score_Y = test_score_Y
        self.sensitive_test = sensitive_test

        self.sensitive_features_dict = sensitive_features_dict
        self.train_size = len(self.train_X)
        self.trained = False
        self.groups = None

    def fit(self, X, y):
        """ No need to implement this as it is already taken care of. We simply need
        to map
        """
        self.answers_map = {}
        for i, sample in enumerate(self.train_X):
            self.answers_map[(sample[0], sample[1])] = (self.sensitive_train[i], self.train_score_Y[i], self.train_Y[i])
        
        if self.test_X is not None:
            for i, sample in enumerate(self.test_X):
                self.answers_map[(sample[0], sample[1])] = (self.sensitive_test[i], self.test_score_Y[i], self.test_Y[i])
        pass

    def predict(self, samples, sensitive_features=None):
        # predict the outcome of the model on the given samples. If samples
        # is none, then the self.test_data will be used
        # return the predictions scores
        out = np.ones(len(samples))
        for i, sample in enumerate(samples):
            out[i] = self.answers_map[(sample[0], sample[1])][1]
        return out
    
    def predict_hard(self, samples, sensitive_features=None):
        # predict the outcome of the model on the given samples. If samples
        # is none, then the self.test_data will be used
        # return the loss as well as the predictions
        scores = np.round(self.predict(samples, sensitive_features))
        return scores

    def get_group_confusion_matrix(self, sensitive_features, X, true_Y):
        # For a trained classifier, get the true positive and true negative rates based on
        # group identity. Dobased on groups (currently only works for binary)
        # sensitive_index is the index of the sensitive attribute.
        #
        # Two returned dictionaries
        groups = np.unique(sensitive_features)

        y_pred_probs = self.predict(X, sensitive_features)
        y_pred = np.round(y_pred_probs)
        micro_acc = 1 - np.sum(np.power(true_Y - y_pred, 2))/len(true_Y)
        print("Overall Accuracy: ", micro_acc)

        micro_auc = roc_auc_score(true_Y, y_pred_probs)
        print("Overall AUC: ", micro_auc)

        macro_acc = 0
        macro_auc = 0

        out_dict = {}   # The format is: {group:[tp, fp, tn, fn]}
        for index, group in enumerate(groups):
            indicies = np.where(sensitive_features==group)[0]
            true_class = true_Y[indicies]
            pred_class = y_pred[indicies]
            
            true_pos_index = np.where(true_class==1)[0]
            true_neg_index = np.where(true_class==0)[0]
            if len(true_pos_index) == 0 or len(true_neg_index) == 0:
                print("No True positives of true negatives in this group")
                continue

            tp = len(np.where(pred_class[true_pos_index]==1)[0])/len(true_pos_index)
            tn = len(np.where(pred_class[true_neg_index]==0)[0])/len(true_neg_index)
            fp = len(np.where(pred_class[true_neg_index]==1)[0])/len(true_neg_index)
            fn = len(np.where(pred_class[true_pos_index]==0)[0])/len(true_pos_index)
            auc = roc_auc_score(true_class, y_pred_probs[indicies])
            macro_auc += auc

            accuracy = 1 - np.sum(np.power(true_class - pred_class, 2))/len(true_class)
            macro_acc += accuracy
            out_dict[group] = [tp, tn, fp, fn, accuracy, auc]
            print(group, "confusion matrix")
            if tp == 0 and fp == 0:
                print("None classified as Positive in group", group)
                print("\t Group Accuracy: ", accuracy)
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2*precision*recall/(precision+recall)
                print("\t F1 score: ", f1)
                print("\t AUC: ", auc)
                print("\t Group Accuracy: ", accuracy)
                print("\t True positive rate:", tp)
                print("\t True negative rate:", tn)
                print("\t False positive rate:", fp)
                print("\t False negative rate:", fn)

        macro_acc /= len(groups)
        macro_auc /= len(groups)

        return out_dict, {"Accuracy": (micro_acc, macro_acc), "AUC": (micro_auc, macro_auc)}


class fair_classifier(pseudo_classifier):
    def __init__(self, train_X, train_y, train_score_y, sensitive_train, \
            test_X, test_y, test_score_y, sensitive_test, metric, sensitive_features_dict=None):
        self.train_X = train_X
        self.train_Y = train_y
        self.train_score_Y = train_score_y
        self.sensitive_train = sensitive_train
        
        self.test_X = test_X
        self.test_Y = test_y
        self.test_score_Y = test_score_y
        self.sensitive_test = sensitive_test
        
        self.sensitive_features_dict = sensitive_features_dict
        self.erm_classifier = pseudo_classifier(self.train_X, self.train_Y, self.train_score_Y, \
                self.sensitive_train, self.test_X, self.test_Y, self.test_score_Y, self.sensitive_test)
        assert(metric in ["equalized_odds", "demographic_parity"])
        self.metric = metric

    def fit(self):
        self.erm_classifier.fit(self.train_X, self.train_Y)
        self.model = ThresholdOptimizer(estimator=self.erm_classifier, constraints=self.metric, prefit=True)
        self.model.fit(self.train_X, self.train_Y, sensitive_features=self.sensitive_train) 

    def predict(self, x_samples, sensitive_features):
        y_samples = self.model.predict(x_samples, sensitive_features=sensitive_features)
        return y_samples
   
    def get_accuracy(self, X, y_true, sensitive_features):
        y_pred = self.predict(X, sensitive_features)
        return 1 - np.sum(np.power(y_pred - y_true, 2))/len(y_true) 

    def predict_prob(self, x_samples, sensitive_features):
        y_samples = self.model._pmf_predict(x_samples, sensitive_features=sensitive_features)
        return y_samples

    def get_avg_group_confusion_matrix(self, sensitive_features, X, true_Y):
        # produces average tp/fp/tn/fn/acc per group
        # Basically get_group_confusion_matrix but modified to return average values where possible
        # For a trained classifier, get the true positive and true negative rates based on
        # group identity. Dobased on groups (currently only works for binary)
        # sensitive_index is the index of the sensitive attribute.
        groups = np.unique(sensitive_features)
        tp_rate = {}
        fp_rate = {}
        tn_rate = {}
        fn_rate = {}

        true_pos_index = np.where(true_Y == 1)
        true_neg_index = np.where(true_Y == 0)

        # Calculate probability of classification for each input
        y_pred_prob = self.predict_prob(X, sensitive_features)
        # Calculate average probability of correct classification (i.e. expected accuracy)
        avg_micro_acc = (np.sum(y_pred_prob[true_pos_index][:,1]) + np.sum(y_pred_prob[true_neg_index][:,0])) / len(true_Y)
        print("Average Overall Accuracy: ", avg_micro_acc)

        micro_auc = roc_auc_score(true_Y, y_pred_prob[:,1])
        print("Overall AUC: ", micro_auc)

        out_dict = {}  # The format is: {group:[tp, fp, tn, fn]}

        avg_macro_acc = 0
        macro_auc = 0

        for index, group in enumerate(groups):
            indicies = np.where(sensitive_features == group)[0]
            true_class = true_Y[indicies]
            pred_prob = y_pred_prob[indicies]

            true_pos_index = np.where(true_class == 1)[0]
            true_neg_index = np.where(true_class == 0)[0]
            if len(true_pos_index) == 0 or len(true_neg_index) == 0:
                print("No True positives or no true negatives in this group")
                continue

            # Find avg rates (i.e. avg probability of tp/tn/fp/fn)
            tp = np.sum(pred_prob[true_pos_index][:,1]) / len(true_pos_index)
            tn = np.sum(pred_prob[true_neg_index][:,0]) / len(true_neg_index)
            fp = np.sum(pred_prob[true_neg_index][:,1]) / len(true_neg_index)
            fn = np.sum(pred_prob[true_pos_index][:,0]) / len(true_pos_index)
            tp_rate[group] = tp
            tn_rate[group] = tn
            fp_rate[group] = fp
            fn_rate[group] = fn

            # Expected accuracy
            accuracy = (np.sum(pred_prob[true_pos_index][:,1]) + np.sum(pred_prob[true_neg_index][:,0])) / len(true_class)
            avg_macro_acc += accuracy

            auc = roc_auc_score(true_class, pred_prob[:,1])
            macro_auc += auc

            out_dict[group] = [tp, tn, fp, fn, accuracy, auc]
            print(group, "average confusion matrix")
            if tp == 0 and fp == 0:
                print("None classified as Positive in group", group)
                print("\t Average Group Accuracy: ", accuracy)
            else:
                # Can't compute F1 out of these since dealing with average values
                #precision = tp / (tp + fp)
                #recall = tp / (tp + fn)
                #f1 = 2 * precision * recall / (precision + recall)
                #print("\t F1 score: ", f1)
                print("\t Average Group Accuracy: ", accuracy)
                print("\t Group AUC: ", auc)
                print("\t Average True positive rate:", tp)
                print("\t Average True negative rate:", tn)
                print("\t Average False positive rate:", fp)
                print("\t Average False negative rate:", fn)

        avg_macro_acc /= len(groups)
        macro_auc /= len(groups)

        return out_dict, {"Accuracy": (avg_micro_acc, avg_macro_acc), "AUC": (micro_auc, macro_auc)}
