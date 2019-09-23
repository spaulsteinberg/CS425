__author__ = 'Samuel Steinberg'
__date__ = 'September 18th, 2019'

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class PredictMPG:

    # pandas options for printing, head for col headers, data is raw, df is dataframe. copy made of dataframe for HP
    def __init__(self, std_option):
        pd.set_option('display.max_rows', 1000)
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        # --                                -- #
        self.__head = ["MPG", "Cylinders", "Displacement", "HP", "Weight", "Acceleration", "Year", "Origin", "Name"]
        self.__data = pd.read_csv('auto-mpg.data', names=self.__head, delimiter="\s+")
        self.__df = pd.DataFrame(data=self.__data)
        self.__option = std_option

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

    # Clean out the questionable data and fill w the mean, delete useless Name field
    def CleanData(self):
        self.__df = self.__df.replace('?', np.nan)
        self.__df['HP'] = self.__df['HP'].astype(float)
        self.__df['HP'] = self.__df['HP'].fillna(self.__df['HP'].mean())
        del self.__df['Name']


    # Call the plots
    def GenPlots(self):
        for ele in self.__head[1:8]:
            self.MakePlot(self.__df[ele], self.__df['MPG'], ele)


    #Create X and r matricies
    def CreateMatricies(self):
        # Make r matrix of MPG vals
        _X = [1] * 398
        vals = []
        for i in self.__df['MPG']:
            vals.append(i)
        self.__r = np.array(vals).reshape(398, 1)

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


    #get standard deviation and get z-scores
    def standardize(self):
        std_vals = []
        indiv_stdev = []
        stdev = 0
        for i in self.__XT[1:]:
            for ele in i:
                stdev += (math.sqrt(pow(abs(ele - i.mean()), 2)) / len(i)) #cumulative stdev
                indiv_stdev.append((math.sqrt(pow(abs(ele - i.mean()), 2)) / len(i)))
            std_vals.append(stdev)
            stdev = 0
        print("\n Standard Deviation Values: ", std_vals)

        # Z-scores
        z_list = []
        j = 0
        for i in self.__XT[1:]:
            for ele in i:
                x = float((ele - i.mean()) / std_vals[j])
                z_list.append(x)
            j += 1
        z_list = np.array(z_list).reshape(7, 398)
        print("\n Z-scores in matrix form: \n", z_list)
        print("\n Z-scores max/min: ", z_list.max(), z_list.min())
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
    pred = PredictMPG(std_orno)
    pred.run()


if __name__ == '__main__':
    main()
