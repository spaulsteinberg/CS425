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
        accuracy = ((conf[0] + conf[3]) / (conf[0] + conf[3] + conf[2] + conf[1]))*100
        TPR = conf[3] / (conf[3] + conf[2])
        PPV = conf[3] / (conf[3] + conf[1])
        TNR = conf[0] / (conf[0] + conf[1])
        F1Score = 2 * PPV * TPR / (PPV + TPR)
        #print("Accuracy: {0:.2f}%".format(accuracy))
        #print("TPR: {0:.2f}%".format(TPR*100))
        #print("PPV: {0:.2f}%".format(PPV*100))
        #print("TNR: {0:.2f}%".format(TNR*100))
        #print("F1 Score: {0:.2f}%".format(F1Score*100))

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
        for i in range(len(self.train_stats)):
            if self.train_stats[k[i]]['Accuracy'] >= most_accurate:
                most_accurate = self.train_stats[k[i]]['Accuracy']
                best_k = k[i]

        # Run best k on test data
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
        self.PlotStats(stats, fields)


    def PlotStats(self, stats, fields):
        fig = plt.figure(figsize=(8, 6))
        plt.gcf().subplots_adjust(bottom=0.40)
        ax = fig.add_subplot(111)
        ax.plot(range(len(fields)), stats, linestyle='-', marker='o')
        plt.xlabel("Metrics")
        plt.ylabel("Percentages")
        plt.title("Best k Performance Graph: k = 6")
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

    def findPurity(self, classification):
        Nmi = 0
        Nm = len(self.train_matrix)
        for ele in range(len(self.train_matrix)):
            if classification == self.train_matrix[ele][-1]:
                Nmi += 1
        return (Nmi/Nm)

    def giniIndex(self):
        return ( (2*self.findPurity(2)) * (1-self.findPurity(4)) )

    def misclassificationError(self):
        return ( 1 - max(self.findPurity(2), (1-self.findPurity(2))) )

    def entropy(self):
        entropy = 0
        entropy += self.findPurity(2) * math.log2(self.findPurity(2))
        entropy += self.findPurity(4) * math.log2(self.findPurity(4))
        return -entropy

    def processUserInput(self, option):
        if option == "entropy": self.node_impurity = self.entropy()
        elif option == "gini": self.node_impurity = self.giniIndex()
        elif option == "misclassification error": self.node_impurity = self.misclassificationError()
        else:
            sys.stderr.write("Invalid option chosen: {}. Quitting...".format(option))
            sys.exit(0)

    # count which classification is most common (has majority) in set
    def majorityClass(self):
        benign_count, malignant_count = 0, 0
        #go through all data in matrix attributes to try and build
        for classification in self.train_matrix[:, -1]:
            if classification == 2: benign_count += 1
            elif classification == 4: malignant_count += 1
        if benign_count > malignant_count or benign_count == malignant_count: return 2
        else: return 4

    def GenerateTree(self):
        #some hard codes here for now
        theta_one = 0.15
        if self.node_impurity < theta_one:
            classification = self.majorityClass()
            ret_node = Node(classification)
            return ret_node
        #else...need to split...


    def runDT(self, option):
        self.train, self.train_matrix, self.validation, self.validation_matrix, self.test, self.test_matrix \
            = self.SplitSet()
        self.processUserInput(option)
        node = self.GenerateTree()
        print(node.left, node.right, node.value)



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
    # perhaps put an option here for what mode
    #kdt.runkNN()
    option = input("Which impurity measure would you like to use (entropy, gini, misclassification error)? ")
    kdt.runDT(option)