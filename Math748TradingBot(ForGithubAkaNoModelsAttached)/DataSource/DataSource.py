from interface import implements, Interface
#from abc import ABCMeta, abstractmethod
#from DataSource import OldDataForModelBuilding
import pandas as pd
import os


class DataSource(Interface):
    """ Serves as the data source interface"""

    def getAskPrice(self):
        """returns the current Ask price as float"""
        raise



    def getRow(self):
        """Returns a Dictionary with the following Keys

        Note: much of the below indicators are specific to meta-trader4

        d['Time'] =          #Time of Tick/Sample
        d['Ask'] =           #Current Ask Price
        d['Volume']=         #Current Trade Volume
        d['ClosingPrice'] =  #Current closing, None if not avaible
        d['Bollinger.Upper.MediumPrice.SD2'] = self.currRow['Bollinger.Upper.MediumPrice.SD2']

        d['RSI(close)'] =    #
        d['MFI'] =           #Money Flow  Index
        d['RSInormalized'] = #
        d['MacDMain'] = self.currRow['MacDMain']#

        d['SMA.weighted'] =  #simpleMovingAverage weighted
        d['Alligator.LipsCurChart'] =
        d['Alligator.Jaw5min']  =
        d['SDPriceValueMedium'] =
        d['stochastic_Fast_Main'] =
        d['stochastic_Fast_Signal'] =
        d['stochastic_Slow_Main'] =
        d['stochastic_Slow_Signal'] =
        d['momentumIndicator'] =
        d['Fractal_Upper'] =
        d['Fractal_Lower'] =
        """
        raise

    def advance(self):
        """For historical/CSV this function will advance to the next row
        For an online data source, this will get contact API for fresh data"""
        pass


#class CleanBrokerData(DataSource):
class CleanBrokerData():
    """So far this cleanBroker Data is getting data from Meta-Trader
        via a csv file.  The file from metaTrader is saved as a csv
        but is not comma seperated, but semi colon seperated
        Thus it converts the semi-colons to commas upon intialization.
        It does this for every update"""
    def __init__(self, path):
        self.path=path
        self.df = None
        self.mostRecentDictStandard = None
        self.dataLocation = -1
        self.currRow= None
        self.advance()



    """Update Runs till there is an actual update being made,
        To update, it goes to path of where meta-trader file is being made
        Once updated the information is turned to a df, then odl csv is deleted
        """
    def advance(self):
        if self.hasNext():
            self.mostRecentDictStandard = self.df.iloc[self.dataLocation]
            #print("Do we get here?", self.dataLocation)
            self.dataLocation += 1
        else:
            on=True
            while(on):
                try:
                    df = pd.read_csv(self.path, sep=';')
                    # for col in df.columns:
                    #     print(col)
                    df['SMacDMain'] = df['Ask'] - df['MacDMain']
                    df['SBollinger.Upper.MediumPrice.SD2.5'] = df['Ask'] - df['Bollinger Upper MediumPrice SD2.5']
                    df['SAlligator.Jaw5min'] = df['Ask'] - df['Alligator Jaw5min']
                    df['SBollinger.Lower.MediumPrice.SD2'] = df['Ask'] - df['Bollinger Lower MediumPrice SD2']
                    df['SBB.LL.SD4'] = df['Ask'] - df['BB LL SD4']
                    df['SSMA.weighted'] = df['Ask'] - df['SMA weighted']
                    df['SAlligator.JawCurChart'] = df['Ask'] - df['Alligator JawCurChart']
                    self.df=df
                    self.removeCSV()#Cleans out CSV file for new one to be made by meta-trader
                    on=False
                    self.dataLocation=1
                    #print("Updated dataFrame from CSV to MLQ4")
                    #print(self.df)#Delete Once I get working
                    #print(self.df.iloc[0])
                    #print(self.standardize_data(self.df.iloc[0]))
                    self.mostRecentDictStandard =  self.df.iloc[0]#self.standardize_data(self.df.iloc[0])
                    #print("most recent dictionary", self.mostRecentDictStandard) #Something wrong is happening here
                except Exception as e:

                    #print("update data failed because, ",  e) #This will print all day
                    pass
        pass



    #TODO implement for SKLEARN
    def getPandasCurrRow(self):
        #return self.currRow
        pass

    def standardize_data(self, currRow):
        d={}
        d['Time'] = currRow['Timestamp'].values[0]         #Time of Tick/Sample
        d['Ask'] =  currRow['Ask'].values[0]      #Current Ask Price
        d['Volume']= 0         #Current Trade Volume
        d['ClosingPrice'] = currRow['Close'].values[0] #Current closing, None if not avaible
        d['RSI(close)'] =  currRow['rsi(Price_Close)'].values[0]#
        d['RSInormalized'] = currRow['Close'].values[0]#
        d['MFI'] = currRow['Money Flow INdex'].values[0]          #Money Flow  Index
        d['SDPriceValueMedium'] =  currRow['standDeviationPriceValueMedium'].values[0]
        d['stochastic_Fast_Main'] = currRow['stochasticArr_Fast_Main'].values[0]
        d['stochastic_Fast_Signal'] = currRow['stochasticArr_Fast_Signal'].values[0]
        d['stochastic_Slow_Main'] = currRow['stochasticArr_Slow_Main'].values[0]
        d['stochastic_Slow_Signal'] = currRow['stochasticArr_Slow_Signal'].values[0]
        d['momentumIndicator'] = currRow['momentumIndicator'].values[0]
        d['Fractal_Upper'] = currRow['Fractal Upper'].values[0]
        d['Fractal_Lower'] = currRow['Fractal Lower'].values[0]
        return d


    def getRow(self):
        return self.mostRecentDictStandard

    def getAskPrice(self):
        return self.mostRecentDictStandard['Ask']

    def removeCSV(self):
        os.remove(self.path)

    def getMostRecent(self):
        return self.df.tail(1)

    def getMostRecentDictStandard(self):
        return self.df.tail(1)

    def getAsk(self):
        df2 = self.df.tail(1)
        return df2['Ask'].sum()

    def hasNext(self):
        if not isinstance(self.df, pd.DataFrame):
            return False
        else:
            return self.dataLocation<len(self.df)-1


class DataClean():
    def __init__(self, path):
        #self.replaceSemiColons(path)
        self.df = pd.read_csv(path, sep=';')
        path=path[:-4]+"commad.csv"
        self.df.to_csv(path, sep=',')


    def replaceSemi(self, path):
        import csv
        with open(path, newline='') as myFile:
            reader = csv.reader(myFile, delimiter=';') #, quoting=csv.QUOTE_NONE
            for row in reader:
                print(row)


    def hasNext(self):
        return True




    def replaceSemiColons(self, path):
        f = open(path, 'a')
        raw = f.read()
        raw = raw.replace(';', ',')
        #f.seek(0)
        f.write(raw)
        #f.truncate()


if __name__ == "__main__":

    #Note to self, I have two tests here, one for cleaning the data I.E. replacing semicolons with commas
    #Then another test below that is for getting data directly from MQL4

    ###Clean Data Test
    #path = 'D:/Data Science DATA Library/OfficialDataUseAsOf11_1_20/Math748DataCollectFinal.csv'
    #test = DataClean(path)

    #result = test.df.head(10)
    #print(result)

    """Clean Broker(Aka Live Data) testing
        Testing Advance and testing standardize dic
    """
    path='C:/Users/garre/AppData/Roaming/MetaQuotes/Terminal/B83207E76A7859F5556693074AFE91E8/MQL4/Files/Data/Math748PriceSending.csv'
    test2 = CleanBrokerData(path=path)
    for i in range(0, 100):
        test2.advance()




