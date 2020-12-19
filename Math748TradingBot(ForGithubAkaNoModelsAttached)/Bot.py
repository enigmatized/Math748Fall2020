import inspect
import time
from datetime import datetime

class Bot:
    def __init__(self, dataSource, broker, featureProviders, delay=2):
        #self.agent          = agent
        self.rewardQ = []
        self.Qdelay = 1

        #liveDataPathSourceForEURSD='C:/Users/garre/AppData/Roaming/MetaQuotes/Terminal/B83207E76A7859F5556693074AFE91E8/MQL4/Files/Data/Math748PriceSending.csv'
        self.mostRecentData = None
        self.broker         = broker
        self.dataSource     = dataSource
        #self.setupDataScoure() #Setups  up the above Nones
        self.TICKSFROMLASTTRADE =0

        self.POWERButton    = True
        self.lastUsedData   = None
        #Just to get an idea
        self.state          = None
        self.previousState  = None
        self.currPrice      = self.dataSource.getAskPrice()


        ########List of Machine learning Features ##########
        self.features       = featureProviders

        #Get State cannot be made till Data source and features are builts
        self.updateNextState()
        self.newPrice = None
        self.lastAction = None
        self.DELAY = delay


    def run(self):
        #Setup Functions
        while self.POWERButton:
            self.dataSource.advance()
            if self.updateNextState():
                #print(self.state)
                if isinstance(self.state[1], tuple):
                    #print(self.state)
                    if sum(list(self.state[1])):#TODO make a difference of what type of buy order being sent in
                        print("BUY SIGNAL SENT", datetime.now())
                        #print(dataSource.getRow())
                        self.takeAction("buy")
                pass
            if self.DELAY != 0:
                time.sleep(self.DELAY)
        pass
    pass


    def takeAction(self, nextAction):
        self.broker.takeAction(nextAction)


    def updateNextState(self):

        newData = {   "TotalBalance": self.broker.getTotalHoldings(),
                        "Ask": self.dataSource.getAskPrice(),
                        "Buy": self.dataSource.getAskPrice()
                        #"Features": self.features.getFeatures(self.dataSource)
        }
        ask = round(self.dataSource.getAskPrice(), 5)
        lotSizeHolding = round(self.broker.getHoldings(), 5)

        #TODO THis is random Forest Machine learning model
        # can you believe RFDirection has a 95% accuracy level.... Outstanding

        #print("Deug print in Class Main at function updateState featureDict: ", featureDict)
        #print(featureDict)
        featureDict = self.features.randomForestModels(self.dataSource)
        try:

            #bigMovein30 = featureDict['RFBigMove']
            #RFDirection = featureDict['RFDirection']
            RF20PosMov = featureDict['RF20PosMov']
            gbm20PosMov = featureDict['gbm20PosMov']
            gbm30PosMov = featureDict['gbm30PosMov']
            newState = (ask, (RF20PosMov, gbm20PosMov, gbm30PosMov))
            #print("Successful prediction Return")
            #newState = (ask, lotSizeHolding, RFDirection, bigMovein30, RF20PosMov)
        except KeyError as e:
            #print("Fail to get prediction", e)
            newState = (ask, lotSizeHolding)
        #directionPerdiction = featureDict['RFDirection']
        #newState = (ask, lotSizeHolding)

        if self.state is None or newState != self.state:
            self.previousState= self.state
            self.state = newState
            #print("updateNextState is a sucess")# Test Print
            return True
        else:
            return False


    #TODO implement, based off of rules /
    # basically note losing X amout of money in a given week/day
    #Note I should abstract this out because I use it in Q-agent and here in Bot.
    #TODO Create test cases for this on specific situations
    def getLegalAction(self):
        actions = ["hold"]
        if self.TICKSFROMLASTTRADE > 50:
            """Note to self, I think I should eliminate shorting for now
            As that adds a dimension 
            interms of rewards and boker emulator........
            """
            #if self.broker.getCash()>self.state[0] and abs(self.broker.getHoldings())<2:#Cannot hold more than 3 contracts at a time
            if self.broker.getCash() > self.state[0] and self.broker.getHoldings()< 2 and self.broker.getHoldings()> 0:  # hold more than 3 and less than zero
                actions.append("sell")
                actions.append("buy")
            #if self.broker.getHoldings()<=-2: #shorting
            if self.broker.getHoldings() <= 0:
                actions.append("buy")
            if self.broker.getHoldings()>=2:
                actions.append("sell")
            #print("Debug Print, Get legal actions ", actions)
        return actions



    def raiseNotDefined(self):
        fileName = inspect.stack()[1][1]
        line = inspect.stack()[1][2]
        method = inspect.stack()[1][3]
        print("*** Something went wrong at: %s at line %s of %s" % (method, line, fileName))
        #sys.exit(1)

#

if __name__ == "__main__":

    """Get Legal Action Test"""
    from BuySell.BrokerMetaTraderEmulator import *
    from DataSource.OldDataForModelBuilding import *
    from Feature.FeatureProvider import *

    dataSource = OldDataForModelBuild()
    broker = BuySellEmulator(dataSource)
    featureProvider = FeatureProvider()
    agent=None
    bot = Bot(agent, dataSource=dataSource, broker=broker, featureProviders=featureProvider, delay=0.2)
    legalActions = bot.getLegalAction()
    assert set(legalActions)  == {"hold", "buy", "sell"}

    broker = BuySellEmulator(dataSource, holding=3)
    bot = Bot(agent, dataSource=dataSource, broker=broker, featureProviders=featureProvider, delay=0.2)
    legalActions = bot.getLegalAction()
    assert set(legalActions) == {"hold", "sell"}

    broker = BuySellEmulator(dataSource, total=0)
    bot = Bot(agent, dataSource=dataSource, broker=broker, featureProviders=featureProvider, delay=0.2)
    legalActions = bot.getLegalAction()
    assert set(legalActions) == {"hold", "sell"}
