import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from joblib import dump, load

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from DataSource import MetaTraderLiveData, OldDataForModelBuilding
# Use numpy to convert to arrays
import numpy as np
# evaluate random forest algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# evaluate random forest algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

#TODO save model and load


class RF20PosMove:

    def __init__(self, data):
        self.dataSource = data
        train_path = "D:/Data Science DATA Library/Math748ProjectFinal/CleanedDataEvenTrain.csv"
        df = pd.read_csv(train_path)
        df = df.dropna()
        # df['SMacDMain'] = df['Ask'] - df['MacDMain']
        # df['SBollinger.Upper.MediumPrice.SD2.5'] = df['Ask'] - df['Bollinger.Upper.MediumPrice.SD2.5']
        # df['SAlligator.Jaw5min'] = df['Ask'] - df['Alligator.Jaw5min']
        # df['SBollinger.Lower.MediumPrice.SD2'] = df['Ask'] - df['Bollinger.Lower.MediumPrice.SD2']
        # df['SBB.LL.SD4'] = df['Ask'] - df['BB.LL.SD4']
        # df['SSMA.weighted'] = df['Ask'] - df['SMA.weighted']
        # df['SAlligator.JawCurChart'] = df['Ask'] - df['Alligator.JawCurChart']
        dct = {}
        ls = list(df.columns)
        i = 0
        for element in ls:
            dct[element] = i
            print(element, i)
            i = i + 1
        self.dct=dct
        self.yy = df.iloc[:, dct['profit20In60TicksClassPlus']].values
        self.xx = df.iloc[:, np.r_[
                                 dct['rsi.Price_Close.'],
                                 dct['MacDMain'],
                                 dct['MACD.diff.HistogramValue'],
                                 dct['standDeviationPriceValueMedium'],
                                 dct['stochasticArr_Fast_Main'],
                                 dct['stochasticArr_Fast_Signal'],
                                 dct['stochasticArr_Slow_Signal'],
                                 dct['stochasticArr_Slow_Main'],
                                 # dct['stochasticArr_Slow_MainLag5min'],
                                 dct['momentumIndicator'],
                                 dct['Fractal.Upper'],
                                 dct['Fractal.Lower'],
                                 dct['SBollinger.Upper.MediumPrice.SD2.5'],
                                 dct['SAlligator.Jaw5min'],
                                 dct['SBollinger.Lower.MediumPrice.SD2'],
                                 dct['SBB.LL.SD4'],
                                 dct['SSMA.weighted'],
                                 dct['SAlligator.JawCurChart']
                       ]].values
        self.dataSource = df
        self.featuresPd = None
        self.featuresNp = None
        self.model      = None
        self.newData    = None
        try:
            self.model = load('D:/Python/Math748TradingBot/Feature/SavedModels/RF20PosMove.joblib')
        except:
            print('No saved, building RFBigMove')
            self.buildModel()


    def getPred(self, dct):
        pass

    #TODO sclae data?
    def scale_bot_data(self, dct):
        pass


    def getPred(self, bbup3, bbdown3, rsi, macd, stochastic, momentum, fracD, fracU):
        self.newData = [bbup3, bbdown3, rsi, macd, stochastic, momentum, fracD, fracU]#This is wrong, not 100% sure how to implement this with numpy and sklearn
        return self.model.predict(self.newData)

    def getPredSimpleTest(self, stochastic, momentum, fracD, fracU):
        self.newData = [stochastic, momentum, fracD, fracU]#This is wrong, not 100% sure how to implement this with numpy and sklearn
        return self.model.predict(self.newData)

    def getPredSimpleTest1(self, x):
        rf_predict_probabilities = self.model.predict_proba(x)
        # print(rf_predict_probabilities)
        # print(np.where(rf_predict_probabilities > 0.80, 1, 0))
        if rf_predict_probabilities[0][1] > 0.85:
            print("rf 20 pos move", rf_predict_probabilities[0][1])
            return [1]
        else:
            return [0]
        #return self.model.predict(x)


    def buildModel(self):
        train_features, test_features, train_labels, test_labels = train_test_split(self.xx, self.yy,
                                                                                    test_size=0.1,
                                                                                    random_state=748)

        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import make_classification
        rfc = RandomForestClassifier(n_jobs=5)
        param_grid = {
            'n_estimators': [1100],
            'criterion': ["gini", "entropy"],
            #'max_depth': [120],
            'max_features' : ["auto", "sqrt", "log2"]
            #'min_samples_leaf': [5]
        }
        scoreFunction = {"recall": "recall", "precision": "precision"}

        CV_rfc = GridSearchCV(estimator=rfc,
                              param_grid=param_grid,
                              scoring=scoreFunction,
                              refit="precision",
                              return_train_score=True,
                              cv=5)

        ###############################
        # Instantiate model with 1000 decision trees
        #rf = RandomForestClassifier(n_estimators=1000, random_state=42)

        CV_rfc.fit(train_features, train_labels)
        y_pred = CV_rfc.predict(test_features)
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        print(confusion_matrix(test_labels, y_pred))
        print(classification_report(test_labels, y_pred))
        print(accuracy_score(test_labels, y_pred))
        print(CV_rfc.best_params_)
        self.model=CV_rfc
        try:
            dump(CV_rfc, 'D:/Python/Math748TradingBot/Feature/SavedModels/RF20PosMove.joblib')
        except:
            print("ALERT: model RFBigMove built successfully, but failed during saving proccess")
        # # define dataset
        # # X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
        # # define the model
        # #model = RandomForestClassifier()
        # #evaluate the model
        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # n_scores = cross_val_score(rf, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        # # report performance
        # print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))





if __name__ == "__main__":
    dataSource = OldDataForModelBuilding.OldDataForModelBuild()
    #print(dataSource.getPandasCurrRow())
    RF = RF20PosMove(dataSource)

