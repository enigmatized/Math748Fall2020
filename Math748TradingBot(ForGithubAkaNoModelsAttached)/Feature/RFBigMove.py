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


class RFfeature:

    def __init__(self, data):
        self.dataSource = data
        self.featuresPd = None
        self.featuresNp = None
        self.model      = None
        self.newData    = None
        try:
            self.model=None
            #self.model = load('D:/Python/Math748TradingBot/Feature/SavedModels/RFBigMove.joblib')
        except:
            print('No saved, building RFBigMove')
            self.buildModel()


    def getPred(self, dct):
        pass


    def getPred(self, bbup3, bbdown3, rsi, macd, stochastic, momentum, fracD, fracU):
        self.newData = [bbup3, bbdown3, rsi, macd, stochastic, momentum, fracD, fracU]#This is wrong, not 100% sure how to implement this with numpy and sklearn
        return self.model.predict(self.newData)

    def getPredSimpleTest(self, stochastic, momentum, fracD, fracU):
        self.newData = [stochastic, momentum, fracD, fracU]#This is wrong, not 100% sure how to implement this with numpy and sklearn
        return self.model.predict(self.newData)

    def getPredSimpleTest1(self, x):
        return self.model.predict(x)


    def buildModel(self):
        # Labels are the values we want to predict
        #labels = np.array(self.dataSource.data['Ask'].values)
        # Remove the labels from the features
        # axis 1 refers to the columns
        #self.features = self.dataSource.data['Ask'].to_numpy()
        # Saving feature names for later use
        feature_list = list(self.dataSource.data.columns)
        # Convert to numpy array
        #self.featuresNp = np.array(self.features)
        y = self.dataSource.data.iloc[:,71].values     #71 in old data frame is profit of 20 in 60 ticks class
        x = self.dataSource.data.iloc[:,41:44].values  # 41 = 'stochasticArr_Slow_Signal', 42 = 'momentumIndicator' 43 = 'Fractal.Upper' 44 =  'Fractal.Lower'
        train_features, test_features, train_labels, test_labels = train_test_split(x, y,
                                                                                    test_size=0.25,
                                                                                    random_state=42)

        ##Scale data
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        train_features = sc.fit_transform(train_features)
        test_features = sc.fit_transform(test_features)

        ######BaseLine
        # The baseline predictions are the historical averages

        ###############################
        # Instantiate model with 1000 decision trees
        rf = RandomForestClassifier(n_estimators=500, random_state=42)
        # Train the model on training data
        # train_labels=train_labels.reshape(-1, 1)
        #print(train_labels)
        #print(type(print(train_labels)))

        rf.fit(train_features, train_labels)
        y_pred = rf.predict(test_features)
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        print(confusion_matrix(test_labels, y_pred))
        print(classification_report(test_labels, y_pred))
        print(accuracy_score(test_labels, y_pred))
        self.model=rf
        try:
            dump(rf, 'D:/Python/TradingBotReinforecementAndre/Feature/SavedModels/RFBigMove.joblib')
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
    RF = RFfeature(dataSource)
    row = dataSource.getRow()
    row2 = dataSource.getPandasCurrRow()


    df = pd.Series(row).T

    print(df)
    print("dataFrame made from dictioinary", type(df))
    print("row from data frame", type(row2))
    subColumnList= ['stochastic_Slow_Signal', 'momentumIndicator', 'Fractal_Upper']#, 'Fractal_Lower']#'stochasticArr_Slow_Signal', 42 = 'momentumIndicator' 43 = 'Fractal.Upper' 44 =  'Fractal.Lower'
    #x = df.iloc[:, 41:44].values#TODO this will work on old data but not work on new data, because the comulmn index are different.
                                #Must chang change to naming convention
    print("DataSeries Size",df.size)
    x = df.loc[subColumnList].values
    print(x)
    print(RF.getPredSimpleTest1(x.reshape(1, -1)))
