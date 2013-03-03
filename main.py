import itertools, numpy, datahandler as dh
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, LinearLayer, FullConnection, FeedForwardNetwork, BiasUnit
from pybrain.tools.validation import Validator
from matplotlib import pyplot as pp

#Input Data
INDICES = 4
DAYS = 10
startdate = '20020101' #YYYYMMDD
enddate = '20130303' #YYYYMMDD

#Neural Network
INPUT = INDICES * DAYS
HIDDEN = 40
OUTPUT = 1

#Training
ITERATIONS = 10
TRAINING = 2200
TESTING = 609
LRATE = 0.01


#Neural Net Functions
def train(net, data):
    trainer = BackpropTrainer(net, data, learningrate=LRATE, momentum=0.9, weightdecay=0.0001)
    #for _ in range(ITERATIONS):
    #    print trainer.train()
    print "Training..."
    return trainer.trainUntilConvergence(maxEpochs=ITERATIONS)

def create_training_data(input):
    data = SupervisedDataSet(INPUT,OUTPUT)
    count = 0
    while count+DAYS+OUTPUT < TRAINING:
        ins = []
        for index in input:
            ins.extend(index[count:count+DAYS])
        outs = input[0][count+DAYS+1:count+DAYS+OUTPUT+1]
        data.addSample(ins, outs)
        count += 1
    return data

def get_output_vals(net, input):
    outputs = []
    count = TRAINING
    while count+DAYS+OUTPUT < TRAINING+TESTING:
        ins = []
        for index in input:
            ins.extend(index[count:count+DAYS])
        outputs.append(net.activate(ins))
        count += OUTPUT
    realouts = []
    for d in outputs:
        realouts.append(list(d))
    return list(itertools.chain(*realouts))

def get_output_dates(dates):
    return dates[TRAINING+DAYS:TRAINING+TESTING-OUTPUT]

def absolute_error(output, target):
    error = 0
    for i in range(0,len(output)):
        error += abs(output[i]-target[i])
    error /= len(output)
    return error


#Main program

def assembleNetwork():
    n = FeedForwardNetwork()
    n.addInputModule(LinearLayer(INPUT,name="in"))
    n.addModule(SigmoidLayer(HIDDEN/4,name="h1"))
    n.addModule(SigmoidLayer(HIDDEN/4,name="h2"))
    n.addModule(LinearLayer(HIDDEN/4,name="h3"))
    n.addModule(LinearLayer(HIDDEN/4,name="h4"))
    n.addModule(BiasUnit(name="b1"))
    n.addModule(BiasUnit(name="b2"))
    n.addOutputModule(SigmoidLayer(OUTPUT,name="out"))

    n.addConnection(FullConnection(n['in'],n['h1']))
    n.addConnection(FullConnection(n['in'],n['h3']))
    n.addConnection(FullConnection(n['h1'],n['h2']))
    n.addConnection(FullConnection(n['h3'],n['h4']))
    n.addConnection(FullConnection(n['h4'],n['out']))
    n.addConnection(FullConnection(n['h2'],n['out']))
    n.addConnection(FullConnection(n['b1'],n['h2']))
    n.addConnection(FullConnection(n['b2'],n['h4']))
    n.sortModules()
    n = n.convertToFastNetwork()

    return n

#Feed-Forward:
net = assembleNetwork()
net.randomize()

#S&P 500
sp500 = dh.DataHandler("%5EGSPC",startdate,enddate)
spdates = sp500.get_dates()
spvalues = sp500.get_values()

#NASDAQ COMPOSITE INDEX
nasdaq = dh.DataHandler("%5EIXIC",startdate,enddate)
nasvals = nasdaq.get_values()

#MAJOR MARKET INDEX
tot = dh.DataHandler("%5EXMI",startdate,enddate)
totvals = tot.get_values()

#NYSE COMPOSITE INDEX
nyse = dh.DataHandler("%5ENYA",startdate,enddate)
nysevals = nyse.get_values()

vals = [spvalues,nasvals,totvals,nysevals]

data = create_training_data(vals)
errors = train(net,data)

sp_predicted = get_output_vals(net, vals)

val = Validator()
print "MSE: ",val.MSE(sp_predicted,spvalues[TRAINING+DAYS+OUTPUT:TRAINING+TESTING])
print "MAE: ",absolute_error(sp_predicted,spvalues[TRAINING+DAYS+OUTPUT:TRAINING+TESTING])

#Configure plots
ENDTRAINING = TRAINING+DAYS
pp.subplot(311)
pp.plot_date(spdates,spvalues,linestyle='solid',c='b',marker='None')
pp.plot_date(spdates[:ENDTRAINING],nasvals[:ENDTRAINING],linestyle='solid',c='g',marker='None')
pp.plot_date(spdates[:ENDTRAINING],totvals[:ENDTRAINING],linestyle='solid',c='c',marker='None')
pp.plot_date(spdates[:ENDTRAINING],nysevals[:ENDTRAINING],linestyle='solid',c='m',marker='None')
pp.plot_date(get_output_dates(spdates),sp_predicted,linestyle='solid',c='r',marker='None')
pp.vlines(spdates[ENDTRAINING],numpy.min(spvalues),numpy.max(spvalues))
pp.xlabel("Date")
pp.ylabel("Normalized Indices")
pp.text(spdates[ENDTRAINING-350],numpy.max(spvalues)*.9,'TRAINING')
pp.text(spdates[ENDTRAINING+50],numpy.max(spvalues)*.9,'PREDICTION')
pp.grid(True)

pp.subplot(313)
pp.plot(numpy.reshape(errors[0],(len(errors[0]),1)))
pp.xlabel("Epoch Number")
pp.ylabel("MSE")
pp.grid(True)

pp.show()