import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer

class pseudo_classifier:
    """ Note that the actual classifier is already trained (unaware as it ought to be) and 
    post-processing does not require access to the underlying features. So this class is 
    basically a wrapper around that prediction so we can pass it nicely onto the 
    fairlearn framework
    """
    def __init__(self, train_X, train_y, score_y, sensitive_train, sensitive_features_dict=None):
        self.train_X = train_X
        self.train_Y = train_y
        self.score_Y = score_y
        self.sensitive_train = sensitive_train

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
            self.answers_map[(sample[0], sample[1])] = (self.sensitive_train[i], self.score_Y[i], self.train_Y[i])
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
        groups = np.unique(sensitive_features)
        n_groups = len(groups)
        tp_rate = {}
        fp_rate = {}
        tn_rate = {}
        fn_rate = {}
        
        y_pred = self.predict_hard(X, sensitive_features)
        accuracy = 1 - np.sum(np.power(true_Y - y_pred, 2))/len(true_Y) 
        print("Overall Accuracy: ", accuracy)

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
            tp_rate[group] = tp 
            tn_rate[group] = tn
            fp_rate[group] = fp 
            fn_rate[group] = fn 
           
            accuracy = 1 - np.sum(np.power(true_class - pred_class, 2))/len(true_class) 
            print(group, "confusion matrix")
            if tp == 0 and fp == 0:
                print("None classified as Positive in group", group)
                print("\t Group Accuracy: ", accuracy)
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2*precision*recall/(precision+recall)
                print("\t F1 score: ", f1)
                print("\t Group Accuracy: ", accuracy)
                print("\t True positive rate:", tp)
                print("\t True negative rate:", tn)
                print("\t False positive rate:", fp)
                print("\t False negative rate:", fn)
        
        return tp_rate, fp_rate, tn_rate, fn_rate


class fair_classifier(pseudo_classifier):
    def __init__(self, train_X, train_y, score_y, sensitive_train, \
            metric, sensitive_features_dict=None):
        self.train_X = train_X
        self.train_Y = train_y
        self.score_Y = score_y
        self.sensitive_train = sensitive_train
        self.sensitive_features_dict = sensitive_features_dict
        self.erm_classifier = pseudo_classifier(self.train_X, self.train_Y, self.score_Y, \
                self.sensitive_train)
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
    

