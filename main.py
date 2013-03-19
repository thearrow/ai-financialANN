import itertools, numpy, datahandler as dh
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, LinearLayer, FullConnection, FeedForwardNetwork, BiasUnit
from pybrain.tools.validation import Validator
from matplotlib import pyplot as pp

#Input Data
INDICES = 1
DAYS = 4
startdate = '20020101'  # YYYYMMDD

#Neural Network
INPUT = INDICES * DAYS + INDICES * 2
HIDDEN = 30
OUTPUT = 1

#Training
ITERATIONS = 20
TRAINING = 2200
TESTING = 613
LRATE = 0.8
MOMENTUM = 0.4


#Neural Net Functions
def train(net, data):
    trainer = BackpropTrainer(net, data, learningrate=LRATE, momentum=MOMENTUM, weightdecay=0.0001)
    print "Training..."
    #for _ in range(ITERATIONS):
    #    print trainer.train()
    return trainer.trainUntilConvergence(maxEpochs=ITERATIONS)


#Main program
def assembleNetwork():
    # (INPUT : HIDDEN : HIDDEN : OUTPUT)
    n = FeedForwardNetwork()
    n.addInputModule(LinearLayer(INPUT, name="in"))
    n.addModule(SigmoidLayer(HIDDEN, name="h1"))
    n.addModule(SigmoidLayer(HIDDEN, name="h2"))
    n.addModule(BiasUnit(name="bias"))
    n.addOutputModule(SigmoidLayer(OUTPUT, name="out"))

    n.addConnection(FullConnection(n['bias'], n['in']))
    n.addConnection(FullConnection(n['in'], n['h1']))
    n.addConnection(FullConnection(n['h1'], n['h2']))
    n.addConnection(FullConnection(n['h2'], n['out']))
    n.sortModules()
    n = n.convertToFastNetwork()
    return n

#Feed-Forward:
net = assembleNetwork()
net.randomize()

#S&P 500
sp500 = dh.DataHandler()
sp500.load_index("%5EGSPC", startdate)

#NASDAQ COMPOSITE INDEX
#nasdaq = dh.DataHandler()
#nasdaq.load_index("%5EIXIC", startdate)

#MAJOR MARKET INDEX
#tot = dh.DataHandler()
#tot.load_index("%5EXMI", startdate)

#NYSE COMPOSITE INDEX
#nyse = dh.DataHandler()
#nyse.load_index("%5ENYA", startdate)


#Configure plots
#datestr2num
# ENDTRAINING = TRAINING + DAYS
# predicted_dates = get_output_dates(spdates)
# pp.subplot2grid((3, 4), (0, 0), colspan=4)
# pp.plot_date(spdates, spvalues, linestyle='solid', c='black', marker='None', label="S&P500 Actual")
# pp.plot_date(spdates, nasvals, linestyle='dashed', c='g', marker='None', alpha=0.4)
# pp.plot_date(spdates, totvals, linestyle='dashed', c='b', marker='None', alpha=0.4)
# pp.plot_date(spdates, nysevals, linestyle='dashed', c='m', marker='None', alpha=0.4)
# pp.plot_date(predicted_dates, sp_predicted, linestyle='solid', c='r', marker='None', label="S&P500 Predicted")
# pp.vlines(spdates[ENDTRAINING], numpy.min(spvalues), numpy.max(spvalues))
# pp.ylabel("Normalized Indices")
# pp.text(spdates[ENDTRAINING - 350], numpy.max(spvalues) * .9, 'TRAINING')
# pp.text(spdates[ENDTRAINING + 50], numpy.max(spvalues) * .9, 'PREDICTION')
# pp.legend(fontsize='small', loc=4)
# pp.grid(True)


# pp.show()