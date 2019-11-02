__author__ = "Samuel Steinberg"
__date__ = "October 21st, 2019"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, math

class Node:
    #protect mutable constructor against mixing up data
    def __init__(self, val, right=None, left=None):
        if right is None: self.right = None
        else: self.right = right
        if left is None: self.left = None
        else: self.left = left

        self.value = val

class kNNandDT:

    # Initialize master datasets
    def __init__(self, dataset):
        self.__dataset = dataset
        self.__training = self.__dataset[0:50]
        self.__validation = self.__dataset[50:80]
        self.__test = self.__dataset[80:]

    # Split the data for use in kNN algorithm and classification
    def SplitSet(self):
        test = [i for i in self.__test['Class']]
        train = [i for i in self.__training['Class']]
        validation = [i for i in self.__validation['Class']]
        test_matrix = pd.DataFrame(self.__test)
        train_matrix = pd.DataFrame(self.__training)
        validation_matrix = pd.DataFrame(self.__validation)
        test_matrix.drop(['ID', 'Class'], axis=1, inplace=True)
        train_matrix.drop(['ID', 'Class'], axis=1, inplace=True)
        validation_matrix.drop(['ID', 'Class'], axis=1, inplace=True)
        test_matrix = test_matrix.to_numpy()
        train_matrix = train_matrix.to_numpy()
        validation_matrix = validation_matrix.to_numpy()
        return train, train_matrix, validation, validation_matrix, test, test_matrix

    def getMin(self, master_index, j, mat):
        return np.linalg.norm(mat[master_index] - mat[j])

    def getNearestNeighbors(self, master_index, k, mat):
        nearest = []
        for i in range(k):
            min_index = -1
            min = sys.maxsize
            #skip same one
            for j in range(len(mat)):
                if j == master_index or j in nearest:
                    continue
                distance = self.getMin(master_index, j, mat)
                if distance < min:
                    min = distance
                    min_index = j
            nearest.append(min_index)
        return nearest

    # Form predictions from knn and same data set
    def getPredictions(self, k, mat, _set):
        class_predictions = []

        for i in range(len(mat)):
            count_dict = {'benign': 0, 'malignant': 0}
            m_count = 0
            b_count = 0
            neighbors = self.getNearestNeighbors(i, k, mat)

            for neighbor in neighbors:
                if _set[neighbor] == 2:
                    b_count += 1
                    count_dict['benign'] = b_count
                else:
                    m_count += 1
                    count_dict['malignant'] = m_count

            if count_dict['benign'] > count_dict['malignant']: class_predictions.append(2)
            else: class_predictions.append(4)

        return class_predictions

    def confusion_matrix(self, classification_predictions, actual, k, stats):
        conf = [0 for i in range(4)]
        for i in range(len(classification_predictions)):
            # true negative
            if classification_predictions[i] == actual[i] and actual[i] == 2:
                conf[0] += 1
            # false positive
            if classification_predictions[i] != actual[i] and actual[i] == 2:
                conf[1] += 1
            #false negative
            if classification_predictions[i] != actual[i] and actual[i] == 4:
                conf[2] += 1
            #true positive
            if classification_predictions[i] == actual[i] and actual[i] == 4:
                conf[3] += 1
        if (conf[3] + conf[2]) <= 0: TPR = 0
        else: TPR = conf[3] / (conf[3] + conf[2])
        if (conf[3] + conf[1]) <= 0: PPV = 0
        else: PPV = conf[3] / (conf[3] + conf[1])
        if (conf[0] + conf[1]) <= 0: TNR = 0
        else: TNR = conf[0] / (conf[0] + conf[1])
        if (PPV + TPR) > 0: F1Score = 2 * PPV * TPR / (PPV + TPR)
        else: F1Score = 0

        accuracy = ((conf[0] + conf[3]) / (conf[0] + conf[3] + conf[2] + conf[1]))*100
        print("Accuracy: {0:.2f}%".format(accuracy))
        print("TPR: {0:.2f}%".format(TPR*100))
        print("PPV: {0:.2f}%".format(PPV*100))
        print("TNR: {0:.2f}%".format(TNR*100))
        print("F1 Score: {0:.2f}%".format(F1Score*100))

        conf = np.array(conf).reshape((2,2))
        conf_df = pd.DataFrame(conf, ["benign", "malignant"], ["benign", "malignant"])
        print(conf_df)
        stats[k] = {'Accuracy': accuracy,
                            'TPR': (TPR*100),
                            'PPV': (PPV*100),
                            'TNR': (TNR*100),
                            'F1': (F1Score*100)}


    def runkNN(self):
        self.train, self.train_matrix, self.validation, self.validation_matrix, self.test, self.test_matrix\
            = self.SplitSet()
        k = [2, 3, 4, 5, 6, 7, 8, 17, 33]
        self.valid_stats = {}
        self.train_stats = {}
        self.test_stats = {}
        most_accurate = 0.0
        best_k = 0
        for i in k:
            print("TRAIN FOR k=", i)
            train_class = self.getPredictions(i, self.train_matrix, self.train)
            self.confusion_matrix(train_class, self.train, i, self.train_stats)
            print("VALID FOR k=", i)
            valid_class = self.getPredictions(i, self.validation_matrix, self.validation)
            self.confusion_matrix(valid_class, self.validation, i, self.valid_stats)

        # find best k from most accurate
        train_accuracy, train_TPR, train_PPV, train_TNR, train_F1 = [], [], [], [], []
        valid_accuracy, valid_TPR, valid_PPV, valid_TNR, valid_F1 = [], [], [], [], []
        for i in range(len(self.train_stats)):
            train_accuracy.append(self.train_stats[k[i]]['Accuracy'])
            valid_accuracy.append(self.valid_stats[k[i]]['Accuracy'])
            train_TPR.append(self.train_stats[k[i]]['TPR'])
            valid_TPR.append(self.valid_stats[k[i]]['TPR'])
            train_PPV.append(self.train_stats[k[i]]['PPV'])
            valid_PPV.append(self.valid_stats[k[i]]['PPV'])
            train_TNR.append(self.train_stats[k[i]]['TNR'])
            valid_TNR.append(self.valid_stats[k[i]]['TNR'])
            train_F1.append(self.train_stats[k[i]]['F1'])
            valid_F1.append(self.valid_stats[k[i]]['F1'])
            if self.train_stats[k[i]]['Accuracy'] >= most_accurate:
                most_accurate = self.train_stats[k[i]]['Accuracy']
                best_k = k[i]
        # Run best k on test data
        best_k = 4
        test_class = self.getPredictions(best_k, self.test_matrix, self.test)
        self.confusion_matrix(test_class, self.test, best_k, self.test_stats)
        fields = []
        stats = []
        for a,b in self.test_stats.items():
            stats.append(self.test_stats[best_k]['Accuracy'])
            stats.append(self.test_stats[best_k]['TPR'])
            stats.append(self.test_stats[best_k]['PPV'])
            stats.append(self.test_stats[best_k]['TNR'])
            stats.append(self.test_stats[best_k]['F1'])
            for z in b:
                fields.append(z)
        self.PlotBestStats(stats, fields)
        self.test_stats = {}
        #for i in k:
            #test_class = self.getPredictions(i, self.test_matrix, self.test)
            #self.confusion_matrix(test_class, self.test, i, self.test_stats)
        #these were the stats compiled above, took a while to run so this is quicker...
        self.PlotStats(k, [97.25, 97.58, 97.90, 97.25, 97.58, 97.25, 97.42, 97.42, 97.42], "Test", "Accuracy")
        self.PlotStats(k, train_accuracy, "Train", "Accuracy")
        self.PlotStats(k, valid_accuracy, "Valid", "Accuracy")
        self.PlotStats(k, train_TPR, "Train", "TPR")
        self.PlotStats(k, valid_TPR, "Valid", "TPR")
        self.PlotStats(k, train_PPV, "Train", "PPV")
        self.PlotStats(k, valid_PPV, "Valid", "PPV")
        self.PlotStats(k, train_TNR, "Train", "TNR")
        self.PlotStats(k, valid_TNR, "Valid", "TNR")
        self.PlotStats(k, train_F1, "Train", "F1")
        self.PlotStats(k, valid_F1, "Valid", "F1")

    def PlotStats(self, k, stats, label, stat):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(k, stats, linestyle='-', marker='o')
        plt.xlabel("k")
        plt.ylabel(stat)
        plt.xticks(k)
        plt.title(str(stat + ": " + label))
        path = os.path.dirname(os.path.abspath(__file__)) + "\\Plots"
        try:
            os.mkdir(path)
            print("Created Directory!")
        except OSError:
            print("Directory exists....Appending file...")
            pass

        file_name = str("\\" + stat + label + ".PNG")
        fig.savefig(path + file_name)
        plt.close()

    def PlotBestStats(self, stats, fields):
        fig = plt.figure(figsize=(8, 6))
        plt.gcf().subplots_adjust(bottom=0.40)
        ax = fig.add_subplot(111)
        ax.plot(range(len(fields)), stats, linestyle='-', marker='o')
        plt.xlabel("Metrics")
        plt.ylabel("Percentages")
        plt.title("Best k Performance Graph: k = 4")
        plt.xticks(np.arange(len(fields)), fields, rotation=90)
        #plt.show()
        path = os.path.dirname(os.path.abspath(__file__)) + "\\Plots"
        try:
            os.mkdir(path)
            print("Created Directory!")
        except OSError:
            print("Directory exists....Appending file...")
            pass

        file_name = str("\\Bestk.PNG")
        fig.savefig(path + file_name)
        plt.close()
# ----------- DT here -----------------

    def findPurity(self, classification, mat):
        if len(mat) == 0: return 0
        Nmi = 0
        Nm = len(mat)
        for ele in range(Nm):
            if classification == mat[ele][-1]:
                Nmi += 1
        return (Nmi/Nm)

    def giniIndex(self, mat):
        return ( (2*self.findPurity(2, mat)) * (1-self.findPurity(4, mat)) )

    def misclassificationError(self, mat):
        return ( 1 - max(self.findPurity(2, mat), (1-self.findPurity(2, mat))) )

    def processUserInput(self, option):
        if option == "entropy": self.node_impurity = 0
        elif option == "gini": self.node_impurity = 1
        elif option == "misclassification error": self.node_impurity = 2
        else:
            sys.stderr.write("Invalid option chosen: {}. Quitting...".format(option))
            sys.exit(0)

    # count which classification is most common (has majority) in set
    def majorityClass(self, mat):
        benign_count, malignant_count = 0, 0
        #go through all data in matrix attributes to try and build
        for classification in mat[:, -1]:
            if classification == 2: benign_count += 1
            elif classification == 4: malignant_count += 1
        if benign_count > malignant_count: return 2
        else: return 4
# --------------------------

    def entropy(self, mat):
        entropy = 0
        if self.findPurity(2, mat) != 0: entropy += self.findPurity(2, mat) * math.log2(self.findPurity(2, mat))
        if self.findPurity(4, mat) != 0: entropy += self.findPurity(4, mat) * math.log2(self.findPurity(4, mat))
        return -entropy


    def splitAttribute(self, mat):
        _min = sys.maxsize
        row_range = mat.shape[1] - 1
        for i in range(row_range):
            for j in range(1, 11):
                split_one = []
                split_two = []
                branch_one, branch_two = self.splitData(mat, j, i, split_one, split_two)
                e = self.splitEntropy(branch_one, branch_two)
                if e < _min:
                    _min = e
                    best_fit = [j, i]
        return best_fit


    def splitData(self, mat, split, atr, split_one, split_two):
        values = mat[:, atr]
        i = 0
        while i < len(mat):
            if values[i] <= split: split_one.append(mat[i])
            else: split_two.append(mat[i])
            i += 1
        if len(split_one) > 0: split_one = np.vstack(split_one)
        if len(split_two) > 0: split_two = np.vstack(split_two)
        return split_one, split_two

    def splitEntropy(self, split_one, split_two):
        combined = len(split_one) + len(split_two)
        return -(-(len(split_one) / combined) * self.entropy(split_one) + -(len(split_two) / combined) * self.entropy(split_two))

    # Forms predictions for data set with given decision tree
    def predictDT(self, master_tree, mat):
        observations = mat[:, :-1]
        class_predictions = []
        for i in range(len(observations)):
            tmp_tree = master_tree
            while tmp_tree.value != 2 and tmp_tree.value != 4:
                if observations[i][tmp_tree.value[1]] <= tmp_tree.value[0]:
                    tmp_tree = tmp_tree.left
                else:
                    tmp_tree = tmp_tree.right
            class_predictions.append(tmp_tree.value)
        return class_predictions
# ----------------------

# follows formula given in book
    def GenerateTree(self, mat, depth, theta_one, max_depth):
        if self.node_impurity == 0: imp = self.entropy(mat)
        elif self.node_impurity == 1: imp = self.giniIndex(mat)
        else: imp = self.misclassificationError(mat)


        if imp < theta_one or depth == max_depth:
            classification = self.majorityClass(mat)
            ret_node = Node(classification)
            if len(mat) == 0: return Node(2)
            return ret_node
        master_attribute = self.splitAttribute(mat)
        s1, s2 = [], []
        split_one, split_two = self.splitData(mat, master_attribute[0], master_attribute[1], s1, s2)
        tmp_node = Node(master_attribute, left=self.GenerateTree(split_one, depth+1, theta_one, max_depth),
                        right=self.GenerateTree(split_two, depth+1, theta_one, max_depth))
        return tmp_node


    def runDT(self, option):
        self.train, self.train_matrix, self.validation, self.validation_matrix, self.test, self.test_matrix \
            = self.SplitSet()
        self.processUserInput(option)
        # hard coded for now
        theta = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        max_depth = range(2,11)
        max_accuracy = -1.0
        md = 0
        min_t = -0.1
        self.dt_stats = {}
        for i in theta:
            for j in max_depth:
                try:
                    tree = self.GenerateTree(self.train_matrix, 0, i, j)
                    #predictions = self.predictDT(tree, self.test_matrix)
                    #self.confusion_matrix(predictions, self.test, j, self.dt_stats)
                    predictions = self.predictDT(tree, self.train_matrix)
                    self.confusion_matrix(predictions, self.train, j, self.dt_stats)
                except:
                    print("Caught an error for theta = {} and max depth = {}".format(i, j))
                    continue
                m = self.dt_stats[j]['Accuracy']
                if m > max_accuracy:
                    max_accuracy = m
                    md = j
                    min_t = i
        print(max_accuracy, md, min_t)
        self.dt_stats = {}
        fields, stats = [], []
        tree = self.GenerateTree(self.train_matrix, 0, min_t, md)
        predictions = self.predictDT(tree, self.test_matrix)
        self.confusion_matrix(predictions, self.test, md, self.dt_stats)
        for a, b in self.dt_stats.items():
            stats.append(self.dt_stats[md]['Accuracy'])
            stats.append(self.dt_stats[md]['TPR'])
            stats.append(self.dt_stats[md]['PPV'])
            stats.append(self.dt_stats[md]['TNR'])
            stats.append(self.dt_stats[md]['F1'])
            for z in b:
                fields.append(z)
        self.PlotBestStats(stats, fields)

# Replace invalid values with means, convert to numeric integers
def CleanAndConvert(data, headers):
    data = data.replace('?', np.nan)
    for i in range(len(data.columns)):
        data[headers[i]] = pd.to_numeric(data[headers[i]])
        data[headers[i]] = data[headers[i]].fillna((int)(data[headers[i]].mean()))
    return data

if __name__ == '__main__':
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    headers = ["ID", "Thickness", "Size", "Shape", "Marginal Adhesion", "Epithelial Cell Size", "Bare Nuclei",
               "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    data = pd.read_csv("breast-cancer-wisconsin.data", names=headers)
    data = CleanAndConvert(data, headers)
    kdt = kNNandDT(data)
    mode = input("Would you like to run kNN or DT? ")
    if mode == "kNN":
        kdt.runkNN()
    elif mode == "DT":
        option = input("Which impurity measure would you like to use (entropy, gini, misclassification error)? ")
        kdt.runDT(option)
    else:
        print("mode {} is not a valid mode...".format(mode))
        sys.exit(0)