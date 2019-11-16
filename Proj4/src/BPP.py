

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from math import exp

class BPP:
    acc, tnr, tpr, ppv, F1, ll, ep = [], [], [], [], [], [], []
    err_vals = {}
    def __init__(self, f, l, e, h, w):
        np.random.seed(1)
        self.folds = f
        self.learning_rate = l
        self.epochs = e
        self.hidden = h
        self.writer = w
        self.cleanData()

    def cleanData(self):
        self.dataset = []
        with open('spambase.data', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if not row: continue
                self.dataset.append(row)
        dl = len(self.dataset[0]) - 1
        for i in range(dl):
            for row in self.dataset:
                row[i] = float(row[i].strip())
        classification = [row[dl] for row in self.dataset]
        unique = set(classification)
        ltable = {}
        for i, value in enumerate(unique): ltable[value] = i
        for row in self.dataset: row[dl] = ltable[row[dl]]
        self.normalize([[min(column), max(column)] for column in zip(*self.dataset)])



    def normalize(self,data):
        for row in self.dataset:
            for i in range(len(row) - 1):
                row[i] = (row[i] - data[i][0]) / (data[i][1] - data[i][0])

    def crossSplit(self):
        dataset_split = []
        dataset_copy = list(self.dataset)
        fold_size = int(len(self.dataset) / self.folds)
        i = 0
        while i < self.folds:
            fold = list()
            while len(fold) < fold_size:
                index = np.random.randint(0, len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
            i += 1
        return dataset_split


    def confusion_matrix(self):
        conf = [0 for i in range(4)]
        for i in range(len(self.predictions)):
            # true negative
            if self.predictions[i] == self.actual[i] and self.actual[i] == 0:
                conf[0] += 1
            # false positive
            if self.predictions[i] != self.actual[i] and self.actual[i] == 1:
                conf[1] += 1
            # false negative
            if self.predictions[i] != self.actual[i] and self.actual[i] == 0:
                conf[2] += 1
            # true positive
            if self.predictions[i] == self.actual[i] and self.actual[i] == 1:
                conf[3] += 1
        if (conf[3] + conf[2]) <= 0:
            TPR = 0
        else:
            TPR = conf[3] / (conf[3] + conf[2])
        if (conf[3] + conf[1]) <= 0:
            PPV = 0
        else:
            PPV = conf[3] / (conf[3] + conf[1])
        if (conf[0] + conf[1]) <= 0:
            TNR = 0
        else:
            TNR = conf[0] / (conf[0] + conf[1])
        if (PPV + TPR) > 0:
            F1Score = 2 * PPV * TPR / (PPV + TPR)
        else:
            F1Score = 0

        accuracy = ((conf[0] + conf[3]) / (conf[0] + conf[3] + conf[2] + conf[1])) * 100
        print("Accuracy: {0:.2f}%".format(accuracy))
        print("TPR: {0:.2f}%".format(TPR * 100))
        print("PPV: {0:.2f}%".format(PPV * 100))
        print("TNR: {0:.2f}%".format(TNR * 100))
        print("F1 Score: {0:.2f}%".format(F1Score * 100))
        conf = np.array(conf).reshape((2, 2))
        conf_df = pd.DataFrame(conf, ["Not spam", "spam"], ["Not spam", "spam"])
        print(conf_df)
        self.acc.append(accuracy/100)
        self.tpr.append(TPR)
        self.tnr.append(TNR)
        self.ppv.append(PPV)
        self.F1.append(F1Score)
        self.ll.append(self.learning_rate)
        self.ep.append(self.epochs)
        # ----- Used for writing data to csv ------- #
        #header = [("Folds:", self.folds), ("Learning:", self.learning_rate), ("Epochs:", self.epochs),
         #         ("Hidden:", self.hidden)]
        #data = [("Accuracy", accuracy), ("TPR", TPR), ("PPV", PPV), ("TNR", TNR), ("F1 Score", F1Score)]
        #header = zip(*header)
        #for head in header:
        #    self.writer.writerow(head)
        #for ele in data:
        #    self.writer.writerow(ele)
        #self.writer.writerows("\n\n")


    def train_test(self, test, fold, train, algorithm, *args):
        for row in fold:
            row_copy = list(row)
            test.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train, test, *args)

    def eval(self, algorithm, *args):
        folds = self.crossSplit()
        for fold in folds:
            train = list(folds)
            train.remove(fold)
            train = sum(train, [])
            test = []
            self.train_test(test, fold, train, algorithm, *args)
            self.actual = [row[-1] for row in fold]
        self.confusion_matrix()



    def activation(self, weights, inputs):
        self.activate = weights[-1]
        for i in range(len(weights) - 1):
            self.activate += weights[i] * inputs[i]



    def transfer(self):
        return 1.0 / (1.0 + exp(-self.activate))


# Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                self.activation(neuron['weights'], inputs)
                neuron['output'] = self.transfer()
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def transfer_derivative(self, output):
        return output * (1.0 - output)



    def back_prop_err(self, expected, epoch):

        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])

            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])
                self.err_vals.setdefault(epoch, []).append(errors[j] * self.transfer_derivative(neuron['output']))



    def up_weights(self, row):
        i = 0
        while i < len(self.network):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.learning_rate * neuron['delta']
            i += 1


    def train_net(self, train, n_outputs):
        epoch = 0
        while epoch < self.epochs:
            for row in train:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                self.writer.writerows("Epoch {}".format(epoch))
                self.back_prop_err(expected, epoch)
                self.up_weights(row)
            epoch += 1


# Initialize a network
    def init_net(self, n_inputs, n_outputs):
        self.network = []
        self.network.append([{'weights': [np.random.uniform() for i in range(n_inputs + 1)]} for i in range(self.hidden)])
        self.network.append([{'weights': [np.random.uniform() for i in range(self.hidden + 1)]} for i in range(n_outputs)])


    def predict(self,row):
        return self.forward_propagate(row).index(max(self.forward_propagate(row)))


    def back_prop(self, train, test):
        n_inputs = len(train[0]) - 1
        n_outputs = len(set([row[-1] for row in train]))
        self.init_net(n_inputs, n_outputs)
        self.train_net(train, n_outputs)
        self.predictions = []
        for row in test:
            prediction = self.predict(row)
            self.predictions.append(prediction)

    def plotStats(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.ll, self.acc)
        plt.xlabel(str("Learning Rate ({}) epochs".format(self.epochs)))
        plt.ylabel("Accuracy")
        plt.title(str("Learning rate vs. Accuracy"))
        path = os.path.dirname(os.path.abspath(__file__)) + "\\Plots"
        try:
            os.mkdir(path)
            print("Directory successfully created...")
        except OSError:
            print("Directory already exists, appending...")
        fig.savefig(str(path + "\\" + "LRAccuracy.PNG"))
        plt.close()

    def run(self):
        self.eval(self.back_prop)


if __name__ == '__main__':
    folds = [3,4,5,6]
    learning = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, .75, .8, .85, .9, 0.95]
    l2 = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    epochs = [5, 10, 20, 30]
    with open("temp_test.csv", 'w+', newline='') as f:
        wr = csv.writer(f, delimiter=",")
        for learn in l2:
            backprop = BPP(5, learn, 5, 5, wr)
            backprop.run()
    f.close()
