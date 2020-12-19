

import DataSource
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing


class MTLiveData:
    def __init__(self):
        self.data = pd.read_csv('C:/Users/garre/AppData/Roaming/MetaQuotes/Terminal/B83207E76A7859F5556693074AFE91E8/MQL4/Files/Data/liveData.csv')
        self.profitLossStats = pd.read_csv('C:/Users/garre/AppData/Roaming/MetaQuotes/Terminal/B83207E76A7859F5556693074AFE91E8/MQL4/Files/Data/liveData.csv')



    def update(self):
        newData = pd.read_csv('C:/Users/garre/AppData/Roaming/MetaQuotes/Terminal/B83207E76A7859F5556693074AFE91E8/MQL4/Files/Data/liveData.csv')
        newData.drop_duplicates()
        if(self.data.equals(newData)):
            pass
        else:
            self.data = newData

    def clean(self, data):
        data.drop_duplicates()







