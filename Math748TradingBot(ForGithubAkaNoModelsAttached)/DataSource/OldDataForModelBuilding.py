import DataSource
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from abc import ABCMeta, abstractmethod
from interface import implements, Interface
from DataSource import DataSource as DataSource
#class OldDataForModelBuild(implements(DataSource.DataSource)):
class OldDataForModelBuild():
    def __init__(self):
        self.data = pd.read_csv('C:/Users/garre/OneDrive/Documents/cleanDat748_10_4_20.csv')
        self.data.info()
        # print(data.shape) For note books really
        self.dataLocation = -1
        self.currRow= None
        self.advance()
        #self.df =None

    #TODO note this is for loaded/old training data/
    # Notsure if this should be getNextRow
    # Or get curr road
    # this could be a problem in terms of when calling update
    # this is a form of update
    # TODO SUPER: make the update for everything more encapsulate.
    # TODO draw a diagram for this
    #Not sure


    def getAskPrice(self):
        return self.currRow['Ask']

    def getRow(self):
        d={}
        d['Time'] = self.currRow['Timestamp']         #Time of Tick/Sample
        d['Ask'] =  self.currRow['Ask']         #Current Ask Price
        d['Volume']= 0         #Current Trade Volume
        d['ClosingPrice'] = self.currRow['Close'] #Current closing, None if not avaible
        d['Bollinger.Upper.MediumPrice.SD2'] = self.currRow['Bollinger.Upper.MediumPrice.SD2']
        d['RSI(close)'] =  self.currRow['rsi.Price_Close.']#
        d['MFI'] = self.currRow['Money.Flow.INdex']
        d['SMA.weighted'] = self.currRow['SMA.weighted']
        d['MacDMain'] = self.currRow['MacDMain']#
        d['Alligator.LipsCurChart'] = self.currRow['Alligator.LipsCurChart']
        d['Alligator.Jaw5min'] = self.currRow['Alligator.Jaw5min']
        d['RSInormalized'] = self.currRow['Close']#
              #Money Flow  Index
        d['SDPriceValueMedium'] =  self.currRow['standDeviationPriceValueMedium']
        d['stochastic_Fast_Main'] = self.currRow['stochasticArr_Fast_Main']
        d['stochastic_Fast_Signal'] = self.currRow['stochasticArr_Fast_Signal']
        d['stochastic_Slow_Main'] = self.currRow['stochasticArr_Slow_Main']
        d['stochastic_Slow_Signal'] = self.currRow['stochasticArr_Slow_Signal']
        d['momentumIndicator'] = self.currRow['momentumIndicator']
        d['Fractal_Upper'] = self.currRow['Fractal.Upper']
        d['Fractal_Lower'] = self.currRow['Fractal.Lower']

        return d


        #return self.currRow.to_dict()


    def advance(self):
        if self.hasNext():
            self.dataLocation+=1
            self.currRow = self.data.iloc[self.dataLocation]
            #self.currRow = self.data.iloc[[self.dataLocation]].to_dict()
        return self.currRow

    def hasNext(self):
        return self.dataLocation<len(self.data)-1

    def getPandasCurrRow(self):
        return self.currRow


if __name__ == "__main__":
    test = OldDataForModelBuild()
    while test.hasNext():
        test.update()
        print(test.getAsk())







