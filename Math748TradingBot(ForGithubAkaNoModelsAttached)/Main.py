from Bot import *


from BuySell.BrokerMetaTraderEmulator import BuySellEmulator
from DataSource.DataSource import CleanBrokerData
from Feature.FeatureProvider import FeatureProvider
from BuySell.BrokerMetaTraderEmulator import BuySellEmulator
from DataSource import MetaTraderLiveData, OldDataForModelBuilding
import sys
from Dprint import *




if __name__ == "__main__":
    mode = sys.argv[1]
    dprint(mode)
    file = sys.argv[2]
    t_epsilon = float(sys.argv[3])
    t_alpha = float(sys.argv[4])
    t_gamma = float(sys.argv[5])
    t_data = sys.argv[6]

    if mode == "training":
        dataSource = OldDataForModelBuilding.OldDataForModelBuild()
        #[print(dataSource.getRow()) for i in range(1, 4)]
        broker     = BuySellEmulator(datasource=None, mode=mode) #BrokerMetaTraderEmulator()
        features   = FeatureProvider()
        agent      = QAgent.QAgent(model=file, epsilon=t_epsilon, alpha=t_alpha, gamma=t_gamma, features=features)
        delay=0
    elif mode == "trainingLive":
        dataSource = CleanBrokerData('C:/Users/garre/AppData/Roaming/MetaQuotes/Terminal/B83207E76A7859F5556693074AFE91E8/MQL4/Files/Data/Math748PriceSending.csv')
        #NOTE AS OF NOVEMBER I CHANGE DATASOURCE OBJECT to be passed into emulator and broker
        #I have not thought that through much and might be a design choice later on
        time.sleep(1)#experiment?
        broker     = BuySellEmulator(dataSource) #BrokerMetaTraderEmulator()
        features   = FeatureProvider()
        #agent      = QAgent.QAgent(model=file, epsilon=t_epsilon, alpha=t_alpha, gamma=t_gamma, features=features)
        delay=0
    elif mode == "testing":
        dataSource = None#TODO this later
        broker     = None#TODO add this later
        agent = QAgent(model=file, epsilon=epsilon, alpha=alpha, gamma=gamma)

    b = Bot(dataSource=dataSource, broker =broker, featureProviders=features, delay=delay)
    #b = Bot(mode)
    try:
        b.run()
    except KeyboardInterrupt:
        print("Closing program, saving model")
        agent.saveModel()


