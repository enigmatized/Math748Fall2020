import inspect
from datetime import time
from DataSource.DataSource import CleanBrokerData
import time
from BuySell import BrokerInterface
from DataSource import OldDataForModelBuilding

class BuySellEmulator(BrokerInterface):
    """Because this is a emulator we can specific beginnning holdings/price"""

    def __init__(self, datasource, mode="trainingLive",  total=3000, holding=0):
        self.total=total
        self.orderStatus = None
        self.COSTofTrade = 0.00004
        self.holding=holding
        if mode == "trainingLive":
            #self.brokerDataSource= CleanBrokerData(path='C:/Users/garre/AppData/Roaming/MetaQuotes/Terminal/B83207E76A7859F5556693074AFE91E8/MQL4/Files/Data/Math748PriceSending.csv')
            self.brokerDataSource=datasource
            datasource.advance()
            print(self.brokerDataSource.getAskPrice())
        elif mode == "training":
            self.brokerDataSource = OldDataForModelBuilding.OldDataForModelBuild()
            print(type(self.brokerDataSource))
        self.lastPrice = self.brokerDataSource.getAskPrice()


    def getHoldings(self):
        #Todo Returns cash on hand
        return self.holding

    def getCash(self):
        #Todo Returns cash on hand
        return self.total

    #TODO this update needs to be for our holdings to the broker
    def update(self):
        #self.brokerDataSource.update()
        #self.total + ((self.holding if self.holding != 0 else 0) * self.getPrice())
        #print()
        pass

    def getPrice(self):
        if None  !=  self.brokerDataSource.getAskPrice():
            return self.brokerDataSource.getAskPrice()
        else:
            return self.lastPrice

    def getTotalHoldings(self):
        try:
            self.update()
            #print(self.brokerDataSource.df)
        except Exception as e:
            print('update failed in BuySellEmulator-getTotalHoldings()->except ', e)
        #print("Debug print statement at ", inspect.stack()[1][1], "line ", inspect.stack()[1][2], "\nself.total", self.total , " self.holding ", self.holding,"self.getPrice() ", self.getPrice())

        return self.total + ((self.holding if self.holding != 0 else 0)*self.getPrice())

    #THIS IS DEMO SO FAR MQL4 is not making any trades
    def buy(self, type, lotSize):
        """Sending to MQL4"""
        #TODO add end of file path for buy and sell
        try:
            f = open('C:/Users/garre/AppData/Roaming/MetaQuotes/Terminal/B83207E76A7859F5556693074AFE91E8/MQL4/Files/Data/PythonOutput/test.csv', 'w')
            f.write("buy")
            #f.write(type+','+str(lotSize))
            f.close()
        except OSError as e:
            print('Failed Buy Attempted failed as a result of:  ', e.with_traceback())
        """This will need to be changed to real balance and price and order status, but for now"""
        #TODO for future use to make this live, in needs to connect to mql4 broker, which is a different proccess
        self.total-=self.getPrice()*lotSize
        self.total -=self.COSTofTrade
        self.holding+=lotSize
        # try: #TODO this below update would be helpful if I was getting data from the broker, but not the case for emulator
        #     self.update()
        #     #self.total-=self.getPrice()*lotSize
        #     #self.holding+=lotSize
        # except Exception as e:
        #     print('Update Price Failed: ', e.with_traceback())
        # pass

    def sell(self, type, lotSize):
        try:#TODO add end of file path for buy and sell
            f = open('C:/Users/garre/AppData/Roaming/MetaQuotes/Terminal/B83207E76A7859F5556693074AFE91E8/MQL4/Files/Data/test.csv', 'a')
            f.write(type+','+str(-lotSize))
            f.close()
        except OSError as e:
            print('Failed Buy Attempt', e)
        """This will need to be changed to real balance and price and order status, but for now"""
        self.total -= self.COSTofTrade
        self.total += self.getPrice() * lotSize
        self.holding -= lotSize
        # try:
        #     self.update()
        # except Exception as e:
        #     print('Update Price Failed: ', e.with_traceback())
        # pass


    #def getPriceActionData(self):

    #No usages of the below function.
    #Delete next time if not needed
    def getBalance(self):
        return 0.0

    #Todo Update action so it is a dictionary or list contatining lot size and type
    # As of now it is hardcoded for EURUSD and lot size of 1
    def takeAction(self, action):
        if action=="buy":
            self.buy("EURUSD", 1)
        elif action=="sell":
            self.sell("EURUSD", 1)
        pass


if __name__ == "__main__":
    test = BuySellEmulator(None, "training")
    test.buy("EURUSD",2)
    #test.sell("EURUSD", 2)
    #print(test.getPrice())
    time.sleep(10)
    #test.update()
    #print(test.getPrice())
