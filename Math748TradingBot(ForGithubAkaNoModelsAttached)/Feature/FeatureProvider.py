import numpy
import pandas as pd

from Feature import RFBigMove
from Feature import Provider
from Feature import RFDirection
from Feature import RF20PosMove
from Feature import GBT20PosMove
from DataSource import MetaTraderLiveData, OldDataForModelBuilding
from Feature.GBT30PosMov import GBT30PosMove


class FeatureProvider(Provider):
    def __init__(self):
        data = OldDataForModelBuilding.OldDataForModelBuild()
        #self.rfBigMove = RFBigMove.RFfeature(data)
        #self.rfDirection = RFDirection.RFfeatureDirection(data)
        self.rf20PosMove = RF20PosMove.RF20PosMove(data)
        self.gbm20PosMove = GBT20PosMove.GBT20PosMove(data)
        self.gbm30PosMove = GBT30PosMove()

    #Should I be passing data to featureProvider or should I have it access?
    #For now I think it is best bot takes care of getting data and passing data to its helpers.


    def getFeatures(self, dataSource):
        dct={}
        #dct['RFBigMove'], dct['RFDirection'] = self.randomForestModels(dataSource)
        #dct['RFBigMove'] = self.randomForestModels(dataSource)
        return dct


    def randomForestModels(self, dataScoure):
        response={}
        #response['RFDirection']= self.randomForestDir(dataScoure)
        #response['RFBigMove']= self.randomForestBigMov(dataScoure)
        response['RF20PosMov'] = self.randomForest20PosMov(dataScoure)
        response['gbm20PosMov'] = self.gbm20PosMovPred(dataScoure)
        response['gbm30PosMov'] = self.gbm30PosMovPred(dataScoure)
        return response

    def gbm20PosMovPred(self, dataScoure):
        row = dataScoure.getRow()
        df = pd.Series(row).T
        subColumnList = ['rsi(Price_Close)',
                           'MacDMain',
                           'MACD-diff-HistogramValue',
                           'standDeviationPriceValueMedium',
                           'stochasticArr_Fast_Main',
                           'stochasticArr_Fast_Signal',
                           'stochasticArr_Slow_Signal',
                           'stochasticArr_Slow_Main',
                           # dct['stochasticArr_Slow_MainLag5min'],
                           'momentumIndicator',
                           'Fractal Upper',
                           'Fractal Lower',
                           'SBollinger.Upper.MediumPrice.SD2.5',
                           'SAlligator.Jaw5min',
                           'SBollinger.Lower.MediumPrice.SD2',
                           'SBB.LL.SD4',
                           'SSMA.weighted',
                           'SAlligator.JawCurChart']
                         #'Fractal Upper', 'Fractal Lower'
        x = df.loc[subColumnList].values
        return self.gbm20PosMove.getPredSimpleTest1(x.reshape(1, -1))[0]

    def gbm30PosMovPred(self, dataScoure):
        row = dataScoure.getRow()
        df = pd.Series(row).T
        subColumnList = ['rsi(Price_Close)',
                           'MacDMain',
                           'MACD-diff-HistogramValue',
                           'standDeviationPriceValueMedium',
                           'stochasticArr_Fast_Main',
                           'stochasticArr_Fast_Signal',
                           'stochasticArr_Slow_Signal',
                           'stochasticArr_Slow_Main',
                           # dct['stochasticArr_Slow_MainLag5min'],
                           'momentumIndicator',
                           'Fractal Upper',
                           'Fractal Lower',
                           'SBollinger.Upper.MediumPrice.SD2.5',
                           'SAlligator.Jaw5min',
                           'SBollinger.Lower.MediumPrice.SD2',
                           'SBB.LL.SD4',
                           'SSMA.weighted',
                           'SAlligator.JawCurChart']
                         #'Fractal Upper', 'Fractal Lower'
        x = df.loc[subColumnList].values
        return self.gbm30PosMove.getPredSimpleTest1(x.reshape(1, -1))[0]






    def randomForest20PosMov(self, dataScoure):
        row = dataScoure.getRow()
        #print("_________________________________________")
        #print(row)
        df = pd.Series(row).T
        # dct = {}
        # ls = list(df.columns)
        # i = 0
        # for element in ls:
        #     dct[element] = i
        #     # print(element, i)
        #     i = i + 1
        # #print("__________________________________________")
        #print(df)
        #print(type(df))
        subColumnList = ['rsi(Price_Close)',
                         'MacDMain',
                         'MACD-diff-HistogramValue',
                         'standDeviationPriceValueMedium',
                         'stochasticArr_Fast_Main',
                         'stochasticArr_Fast_Signal',
                         'stochasticArr_Slow_Signal',
                         'stochasticArr_Slow_Main',
                         # dct['stochasticArr_Slow_MainLag5min'],
                         'momentumIndicator',
                         'Fractal Upper',
                         'Fractal Lower',
                         'SBollinger.Upper.MediumPrice.SD2.5',
                         'SAlligator.Jaw5min',
                         'SBollinger.Lower.MediumPrice.SD2',
                         'SBB.LL.SD4',
                         'SSMA.weighted',
                         'SAlligator.JawCurChart']
                         #'Fractal Upper', 'Fractal Lower'
        x = df.loc[subColumnList].values
        # x = df.iloc[:, numpy.np.r_[
        #                          dct['SMacDMain'],
        #                          dct['SBollinger.Upper.MediumPrice.SD2.5'],
        #                          dct['SAlligator.Jaw5min'],
        #                          dct['SBollinger.Lower.MediumPrice.SD2'],
        #                          dct['SBB.LL.SD4'],
        #                          dct['SSMA.weighted'],
        #                          dct['SAlligator.JawCurChart'],
        #                          dct['rsi.Price_Close.'],
        #                          dct['stochasticArr_Fast_Main'],
        #                          dct['stochasticArr_Fast_Signal'],
        #                          dct['stochasticArr_Slow_Signal'],
        #                          dct['stochasticArr_Slow_Main'],
        #                          # dct['stochasticArr_Slow_MainLag5min'],
        #                          dct['momentumIndicator'],
        #                          dct['standDeviationPriceValueMedium']
        #                      ]].values
        return self.rf20PosMove.getPredSimpleTest1(x.reshape(1, -1))[0]

    def randomForestBigMov(self, dataScoure):
        row = dataScoure.getRow()
        df = pd.Series(row).T
        subColumnList = ['stochastic_Slow_Signal', 'momentumIndicator',
                         'Fractal_Upper']  # , 'Fractal_Lower']#'stochasticArr_Slow_Signal', 42 = 'momentumIndicator' 43 = 'Fractal.Upper' 44 =  'Fractal.Lower'
        x = df.loc[subColumnList].values

        return self.rfBigMove.getPredSimpleTest1(x.reshape(1, -1))[0]


    def randomForestDir(self, dataScoure):
        row = dataScoure.getRow()
        df = pd.Series(row).T
        subColumnList = ['Bollinger.Upper.MediumPrice.SD2', 'RSI(close)', 'MFI', 'MacDMain', 'SMA.weighted',
                         'Alligator.LipsCurChart', 'Alligator.Jaw5min', 'SDPriceValueMedium', 'stochastic_Fast_Main',
                         'stochastic_Fast_Signal', 'stochastic_Slow_Main', 'stochastic_Slow_Signal',
                         'momentumIndicator', 'Fractal_Upper',
                         'Fractal_Lower']

        x = df.loc[subColumnList].values
        return self.rfDirection.getPredSimpleTest1(x.reshape(1, -1))[0]


#TODO create better test cases, I dunno even know what is happening below or why?
if __name__ == "__main__":
    test = FeatureProvider()
    from DataSource import DataSource
    path = 'C:/Users/garre/AppData/Roaming/MetaQuotes/Terminal/B83207E76A7859F5556693074AFE91E8/MQL4/Files/Data/Math748PriceSending.csv'
    new= DataSource.CleanBrokerData(path)
    new.advance()
    print(test.randomForest20PosMov(new))



