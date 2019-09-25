__author__ = 'Samuel Steinberg'
__date__ = 'September 18th, 2019'

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D


class PredictMPG:

    # pandas options for printing, head for col headers, data is raw, df is dataframe. copy made of dataframe for HP
    # option for standardized or not, and number of rows in dataframe
    def __init__(self, std_option):
        pd.set_option('display.max_rows', 1000)
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        # --                                -- #
        self.__head = ["MPG", "Cylinders", "Displacement", "HP", "Weight", "Acceleration", "Year", "Origin", "Name"]
        self.__data = pd.read_csv('auto-mpg.data', names=self.__head, delimiter="\s+")
        self.__df = pd.DataFrame(data=self.__data)
        self.__option = std_option
        self.__nrows = len(self.__df.index)


    #Make simple scatter plot and add line of regression (single)
    def MakePlot(self, X, Y, var1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X, Y, c='r')
        plt.xlabel(var1)
        plt.ylabel("MPG")
        plt.title(str("MPG vs. " + var1))
        plt.grid(True)
        plt.plot(np.unique(X), np.poly1d(np.polyfit(X, Y, 1))(np.unique(X)))
        plt.show()


        # Call the plots
    def GenSinglePlots(self):
        for ele in self.__head[1:8]:
            self.MakePlot(self.__df[ele], self.__df['MPG'], ele)


    # Plotting Z-Scores
    def PlotZscores(self, index, name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(index, self.__mpg_vals, c='r')
        plt.xlabel("Z-scores")
        plt.ylabel("MPG")
        plt.title(str("MPG vs. " + self.__head[name]))
        ax.grid(True)
        path = "C:\\Users\\Sam\\PycharmProjects\\CS425" + "\\Z-Score_Charts\\2D"
        file_name = "\\MPG-" + self.__head[name] + ".PNG"
        try:
            os.mkdir(path)
            print("Created Directory!")
        except OSError:
            print("Directory exists....Appending file...")
            pass
        fig.savefig(path + file_name)
        plt.close()


    # 3D Working --> make like 2D add in for loop w name and vals
    def Plot3DZscores(self, finish, name): #index in, name index, z scores
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Finesse vals to make correct 1D 398 element shape for scatter
        temp= []
        a = []
        for i in self.__df[self.__head[name]]: temp.append(i)
        t_z = self.__z_scores.reshape(1, 7*self.__nrows)
        for i in t_z:
           a = i
        # ----------------------------------------
        start = finish - self.__nrows
        ax.scatter(self.__mpg_vals, temp, a[start:finish], c='y', marker='o')
        ax.set_xlabel("MPG")
        ax.set_ylabel(self.__head[name])
        ax.set_zlabel("Z-Scores")
        ax.w_xaxis.set_pane_color((0.27, 0.15, 0.80, 1.0)) # hot pink -> ((0.90, 0.08, 0.58, 1.0))
        ax.w_yaxis.set_pane_color((0.27, 0.15, 0.80, 1.0))
        ax.w_zaxis.set_pane_color((0.27, 0.15, 0.80, 1.0))
        plt.title(str("MPG v. " + self.__head[name] + " v. Z-Scores"))
        ax.grid(True)
        path = "C:\\Users\\Sam\\PycharmProjects\\CS425" + "\\Z-Score_Charts\\3D"
        file_name = "\\MPG-" + self.__head[name] +"-Z-Scores" + ".PNG"
        try:
            os.mkdir(path)
            print("Created Directory!")
        except OSError:
            print("Directory exists....Appending file...")
            pass
        fig.savefig(path + file_name)

        #plt.show()
        plt.close()


    # Clean out the questionable data and fill w the mean, delete useless Name field
    def CleanData(self):
        self.__df = self.__df.replace('?', np.nan)
        self.__df['HP'] = self.__df['HP'].astype(float)
        self.__df['HP'] = self.__df['HP'].fillna(self.__df['HP'].mean())
        del self.__df['Name']


    #Create X and r matricies
    def CreateMatricies(self):
        # Make r matrix of MPG vals
        _X = [1] * self.__nrows
        self.__mpg_vals = []
        for i in self.__df['MPG']:
            self.__mpg_vals.append(i)
        self.__r = np.array(self.__mpg_vals).reshape(self.__nrows, 1)

        # Make a dataframe copy and insert a new ones col, then delete obsolete MPG and convert to numpy array
        temp_def = self.__df
        temp_def.insert(loc=0, column='Ones', value=_X)
        del temp_def['MPG']
        self.__X = temp_def.to_numpy()
        self.__XT = np.transpose(self.__X)


    # Get the w matrix for coefficients
    def getCoefficients(self):
        # multi-variate equation: w = (X^T (X))^-1(X^T)(r)#
        self.__w = np.dot(np.dot(np.linalg.inv(np.dot(self.__XT, self.__X)), self.__XT), self.__r)
        print("Matrix coefficients: \n", self.__w)


    def getStdev(self):
        self.__std_vals = []
        stdev = 0
        for i in self.__XT[1:]:
            for ele in i:
                stdev += (math.sqrt(pow(abs(ele - i.mean()), 2)) / (1.0*len(i)))  # cumulative stdev
            self.__std_vals.append(stdev)
            stdev = 0
        print("\n Standard Deviation Values: ", self.__std_vals)


# Get Z scores and chart them
    def getZscores(self):
        # Z-scores
        self.__z_scores = []
        j = 0
        for i in self.__XT[1:]:
            for ele in i:
                self.__z_scores.append(float((ele - i.mean()) / self.__std_vals[j]))
            j += 1

        self.__z_scores = np.array(self.__z_scores).reshape(7, self.__nrows)
        name = 1
        index = self.__nrows
        for i in self.__z_scores:
            self.PlotZscores(i, name)
            self.Plot3DZscores(index, name)
            name += 1
            index += self.__nrows
        print("\n Z-scores in matrix form: \n", self.__z_scores)
        print("\n Z-scores max/min: ", self.__z_scores.max(), self.__z_scores.min(), "\n")


    def getStdevMatrix(self):
        # Standardize matricies
        n = []
        x = 0
        for i in self.__XT[1:]:
            for ele in i:
                temp = float((ele - i.mean()) / float(self.__std_vals[x]))
                n.append(temp)
            x += 1
        one = [1] * self.__nrows
        f = one + n
        f = np.array(f).reshape(8, self.__nrows)
        return f

    #get standard deviation and get z-scores
    def standardize(self):
        self.getStdev()
        self.getZscores()
        self.__XT = self.getStdevMatrix()
        self.__X = self.__XT.transpose()
        self.getCoefficients()


    def run(self):
        self.CleanData() # clean bad data and fix columns
        self.CreateMatricies() # Create r, x, x_t matrix
        if self.__option == 0: self.getCoefficients()
        else:
            self.standardize()


def main():
    while(1):
        std_orno = int(input("To standardize enter 1, to not standardize enter 0: "))
        if std_orno != 1 and std_orno != 0: print("Must enter a 1 or 0...")
        else: break
    while(1):
        pred = PredictMPG(std_orno)
        pred.run()
        ra_orno = int(input("To run again, enter a 1. To quit enter a 0..."))
        if ra_orno is 1: continue
        else: break


if __name__ == '__main__':
    main()
