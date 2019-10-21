__author__ = "Samuel Steinberg"

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os, sys

class ReductionAndClustering:
    def __init__(self, df):
        self.__df = df[:57]
        self.__iterations = []
        self.__minInter = []
        self.__maxIntra = []
        self.__DunnIndex = []

    # Clean the dataframe columns and convert them to floats
    def CleanColumns(self, x):
        try:
            if isinstance(x, str):
                x = x.replace("$", "").replace(",", "")
                x = float(x)
            else: return
            return float(x)
        except:
            x = x.strip()
            if x.strip() == "-":
                return float("NaN")
            elif x == '' or x == ' ':
                return float("NaN")


    # Drop utterly useless data and convert from strings to floats
    def CleanData(self):
        self.__colleges = []
        for name in self.__df['Name']:
            self.__colleges.append(name)

        drop_list = ['HBC', 'IPEDS#', '2014 Med School', 'Vet School', '% UG Age 25 +', 'GR Enroll Age 25 +',
                     'ARU Faculty Awards', '% Total Age 25 +', 'Med School Res $', '% UG Pell Grants',
                     'UG Enroll Age 25 +', 'Wall St. Jourl Rank', 'Unnamed: 0', 'Carm R1', 'Name']


        self.__df = self.__df.drop(drop_list, axis=1)
        # ---- Remove dollar signed data stuff, if already a float skip it ---- #
        for i in range(len(self.__df.columns)):
            name = self.__df.columns[i]
            if isinstance(self.__df[name][4], float): continue
            self.__df[name] = self.__df[name].apply(self.CleanColumns)
            self.__df[name] = self.__df[name].fillna(self.__df[name].mean())

        self.__data = self.__df.to_numpy()

    # Extract vals and get graph data
    def ExtractValues(self):
        self.U, self.S, self.V = np.linalg.svd(self.__data)
        _denominator = sum(pow(self.S, 2))
        self.S_vals = []

        # formula S0, S0 + S1, S0+S1+S2,... / sum of all S^2 for each point S
        for i in range(len(self.S)):
            top_sum = 0.0
            for j in range(i+1):
                top_sum += pow(self.S[j], 2)
            self.S_vals.append( ((top_sum / _denominator )*100) )


    #Graph Scree Plot
    def GraphScree(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(self.S)), self.S, 'o-', c='r')
        plt.xlabel("Singular Value Number")
        plt.ylabel("S values")
        plt.title("Scree Graph")
        path = os.path.dirname(os.path.abspath(__file__)) + "\\Plots"
        try:
            os.mkdir(path)
            print("Created Directory!")
        except OSError:
            print("Directory exists....Appending file...")
            pass

        file_name = "\\SingleScree.PNG"
        fig.savefig(path + file_name)
        plt.close()

    # Variance will be the singular values squared and converted to a percentage
    def GraphVariance(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        variance = [(math.pow(i, 2)/10000) for i in self.S_vals]
        ax.plot(range(len(self.S_vals)), variance, 'o-', c='r')
        plt.xlabel("Singular Value Number")
        plt.ylabel("Variance")
        plt.title("Variance Graph")
        path = os.path.dirname(os.path.abspath(__file__)) + "\\Plots"
        try:
            os.mkdir(path)
            print("Created Directory!")
        except OSError:
            print("Directory exists....Appending file...")
            pass

        file_name = "\\SingleVar.PNG"
        fig.savefig(path + file_name)
        plt.close()

    def PCA(self):
        Sigma = np.zeros((self.__data.shape[0], self.__data.shape[1]))
        np.fill_diagonal(Sigma, self.S)
        n_elements = 4 # 1- 51 least to most accurate
        Sigma = Sigma[:, :n_elements]
        self.V = self.V[:n_elements, :]
        # reconstruct
        self.X = self.U.dot(Sigma.dot(self.V))

    def GraphPCA(self, option):
        if option == 1:
            name = "His-Blk"
            fig = plt.figure(figsize=(10,8))
            plt.gcf().subplots_adjust(bottom=0.40)
            ax = fig.add_subplot(111)
            ax.scatter(range(len(self.__colleges)), self.X[:,0], c='orange', label="% Black") #black students
            ax.scatter(range(len(self.__colleges)), self.X[:,1], c='blue', label="% Hispanic") #hispanic students
            plt.xticks(np.arange(len(self.__colleges)), self.__colleges, rotation=90)
            plt.xlabel("Colleges")
            plt.ylabel("PC Value")
            plt.title(str("First PC's"))
            plt.legend()

        elif option == 2:
            name = "US-News-Ranks"
            fig = plt.figure(figsize=(10, 8))
            plt.gcf().subplots_adjust(bottom=0.40)
            ax = fig.add_subplot(111)
            ax.scatter(range(len(self.__colleges)), self.X[:, 2], c='pink', label="US News")
            plt.xticks(np.arange(len(self.__colleges)), self.__colleges, rotation=90)
            plt.xlabel("Colleges")
            plt.ylabel("PC Value")
            plt.title(str("Enrollment"))
            plt.legend()
        else:
            name = "Enrollment"
            fig = plt.figure(figsize=(10, 8))
            plt.gcf().subplots_adjust(bottom=0.40)
            ax = fig.add_subplot(111)
            ax.scatter(range(len(self.__colleges)), self.X[:, 3], c='blue', label="Enrollment")
            plt.xticks(np.arange(len(self.__colleges)), self.__colleges, rotation=90)
            plt.xlabel("Colleges")
            plt.ylabel("PC Value")
            plt.title(str("Enrollment"))
            plt.legend()

        path = os.path.dirname(os.path.abspath(__file__)) + "\\Plots"
        try:
            os.mkdir(path)
            print("Created Directory!")
        except OSError:
            print("Directory exists....Appending file...")
            pass

        file_name = str("\\" + name + ".PNG")
        fig.savefig(path + file_name)
        plt.close()
        if option == 1: self.GraphAlternateTwoPC()

    def GraphAlternateTwoPC(self):
        name = "His-Blk-Compare"
        fig = plt.figure(figsize=(10, 8))
        plt.gcf().subplots_adjust(bottom=0.40)
        ax = fig.add_subplot(111)
        ax.scatter(self.X[:, 0], self.X[:, 1], c='blue')
        for i in range(len(self.__colleges)):
            plt.annotate(i, (self.X[i, 0], self.X[i, 1]))
        plt.xlabel("% Black")
        plt.ylabel("% Hispanic")
        plt.title(str("First Two PC's"))
        path = os.path.dirname(os.path.abspath(__file__)) + "\\Plots"
        try:
            os.mkdir(path)
            print("Created Directory!")
        except OSError:
            print("Directory exists....Appending file...")
            pass

        file_name = str("\\" + name + ".PNG")
        fig.savefig(path + file_name)
        plt.close()

    def clearLists(self):
        self.__DunnIndex.clear()
        self.__maxIntra.clear()
        self.__minInter.clear()
        self.__iterations.clear()

    def kMeansClustering(self, opt, k):
        if opt == 1: _set = self.X
        elif opt == 2: _set = self.X[:,:4]
        else: _set = self.X[:,:2]
        #self.clearLists()
        np.random.seed(200)
        counter = 0
        iteration_cap = 1000 #iteration cap

        m_i = []
        col_max = list(_set.max(axis=0))
        col_min = list(_set.min(axis=0))
        for i in range(k):
            m_i.append([])
            for j in range(len(_set[0])):
                m_i[i].append(np.random.uniform(col_min[j], col_max[j]))

        while counter <= iteration_cap:
            b_i = [ [] for i in range(k) ]

            euclidean = []
            for i in range(len(_set)):
                euclidean.clear()
                for j in range(k):
                    euclidean.append(np.linalg.norm(_set[i] - np.array(m_i[j])))
                    b_i[j].append(0)
                b_i[euclidean.index(min(euclidean))][i] = 1

            m = []
            for i in range(k):
                s = [0 for i in range(_set.shape[1])]
                for j in range(len(_set)):
                    s += b_i[i][j] * _set[j]
                m.append(s / sum(b_i[i]))

            counter += 1
            if np.array_equal(m_i, np.array(m)):
                break
            m_i = m
        print("counter:", counter)
        self.__iterations.append(counter)
        self.getDistancesAndPlot(b_i, _set, k, m_i, opt)


    def getDistancesAndPlot(self, b_i, _set, k, m_i, option):
        if option == 1:
            suf = " All, k = " + str(k)
        elif option == 2:
            suf = " p PC's, k = " + str(k)
        elif option == 3:
            suf = " Two PC's, k = " + str(k)
        clusters = []
        for i in range(k):
            clusters.append([])
            for j in range(len(_set)):
                if b_i[i][j]:
                    clusters[i].append(list(_set[j]))

        # Determine cluster distances
        _min = self.minInterDistance(clusters)
        _max = self.maxIntraDistance(clusters)
        Dunn = _min / _max
        #print(suf)
        #print("Max Intracluster Distance:", _max)
        #print("Min Intercluster Distance:", _min)
        #print("Dunn index:", Dunn)
        self.__minInter.append(_min)
        self.__maxIntra.append(_max)
        self.__DunnIndex.append(Dunn)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(k):
            ax.plot(np.array(clusters[i])[:, 0], np.array(clusters[i])[:, 1], '*')
            ax.scatter(m_i[i][0], m_i[i][1])
        for i in range(len(_set)):
            plt.annotate(i, (self.X[i, 0], self.X[i, 1]))

        plt.title(str("Clustering (k-means)" + suf))
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.grid(linestyle='--', color='lightgray')

        fname = "Clustering" + str(option) + str(k)
        path = os.path.dirname(os.path.abspath(__file__)) + "\\Plots"
        try:
            os.mkdir(path)
            print("Created Directory!")
        except OSError:
            print("Directory exists....Appending file...")
            pass

        file_name = str("\\" + fname + ".PNG")
        fig.savefig(path + file_name)
        plt.close()


    def minInterDistance(self, clusters):
        _min = sys.maxsize
        for i in range(len(clusters)):
            sub_cluster = np.array(clusters[i])
            for j in range(len(sub_cluster)):
                for k in range(len(clusters)):
                    if k == i:
                        continue
                    sub_cluster2 = np.array(clusters[k])
                    for l in range(len(sub_cluster2)):
                        temp = np.linalg.norm(sub_cluster[j] - sub_cluster2[l])
                        if temp < _min:
                            _min = temp
        return _min


    def maxIntraDistance(self, clusters):
        _max = -1
        for i in range(len(clusters)):
            sub_cluster = np.array(clusters[i])

            for j in range(len(sub_cluster)):
                for k in range(len(sub_cluster)):
                    if j == k:
                        continue
                    temp = np.linalg.norm(sub_cluster[j] - sub_cluster[k])
                    if temp > _max:
                        _max = temp
        return _max


    def run(self):
        self.CleanData()
        self.ExtractValues()
        self.GraphScree()
        self.GraphVariance()
        self.PCA()
        self.GraphPCA(1) #minorities
        self.GraphPCA(2) #us news
        self.GraphPCA(3) #enrollment

        #---- FOR LOOP FOR DIFFERENT K-VALS HERE, NOTE AND PLOT EACH ----#
        for i in range(2, 5):
            self.kMeansClustering(1, i) #using whole set
            self.kMeansClustering(2, i) #first 4
            self.kMeansClustering(3, i) #first 2
        #print("Iterations to finish: ", self.__iterations)
        #print("Min dist list", self.__minInter)
        #print("Max dist list", self.__maxIntra)
        #print("Dunn Index list", self.__DunnIndex)

        #make a for loop for clusters and mark the number on the graphs#

def main():
    data = pd.read_csv("UTK-peers.csv")
    df = pd.DataFrame(data=data)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    RC = ReductionAndClustering(df)
    RC.run()

if __name__ == '__main__':
    main()