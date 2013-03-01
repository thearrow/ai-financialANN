import itertools, numpy, datahandler as dh
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from matplotlib import pyplot as pp

#Neural Network
INPUT = 20
HIDDEN = 15
OUTPUT = 1
ITERATIONS = 10
TRAINING = 2000
TESTING = 800
LRATE = 0.05
startdate = '20020101' #YYYYMMDD
enddate = '20130220' #YYYYMMDD


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
    while count+INPUT+OUTPUT < TRAINING:
        ins = list(input[count:count+INPUT])
        outs = input[count+INPUT+1:count+INPUT+OUTPUT+1]
        data.addSample(ins, outs)
        count += 1
    return data

def get_output_vals(net, input):
    outputs = []
    count = TRAINING
    while count+INPUT+OUTPUT < TRAINING+TESTING:
        ins = list(input[count:count+INPUT])
        outputs.append(net.activate(ins))
        count += OUTPUT
    realouts = []
    for d in outputs:
        realouts.append(list(d))
    return list(itertools.chain(*realouts))

def get_output_dates(dates):
    return dates[TRAINING+INPUT:TRAINING+TESTING-OUTPUT]



#Main program

#Recurrent:
#net = buildNetwork(INPUT,HIDDEN,OUTPUT,bias=True,recurrent=True,hiddenclass=LSTMLayer)
#Feed-Forward:
net = buildNetwork(INPUT,HIDDEN,OUTPUT,bias=True)
net.randomize()

#S&P 500
sp500 = dh.DataHandler("%5EGSPC",startdate,enddate)
sp500.normalize()
spdates = sp500.get_dates()
spvalues = sp500.get_values()

#NASDAQ COMPOSITE INDEX
nasdaq = dh.DataHandler("%5EIXIC",startdate,enddate)
nasdaq.normalize()
nasvals = nasdaq.get_values()

#MAJOR MARKET INDEX
tot = dh.DataHandler("%5EXMI",startdate,enddate)
tot.normalize()
totvals = tot.get_values()

#NYSE COMPOSITE INDEX
nyse = dh.DataHandler("%5ENYA",startdate,enddate)
nyse.normalize()
nysevals = nyse.get_values()

data = create_training_data(spvalues)
errors = train(net,data)

sp_predicted = get_output_vals(net, spvalues)


#Configure plots
pp.subplot(311)
pp.plot_date(spdates,spvalues,linestyle='solid',c='b',marker='None')
pp.plot_date(spdates[:TRAINING+INPUT],nasvals[:TRAINING+INPUT],linestyle='solid',c='g',marker='None')
pp.plot_date(spdates[:TRAINING+INPUT],totvals[:TRAINING+INPUT],linestyle='solid',c='y',marker='None')
pp.plot_date(spdates[:TRAINING+INPUT],nysevals[:TRAINING+INPUT],linestyle='solid',c='m',marker='None')
pp.plot_date(get_output_dates(spdates),sp_predicted,linestyle='solid',c='r',marker='None')
pp.vlines(spdates[TRAINING+INPUT],numpy.min(spvalues),numpy.max(spvalues))
pp.xlabel("Date")
pp.ylabel("Normalized Indices")
pp.text(spdates[TRAINING+INPUT-350],numpy.max(spvalues)-50,'TRAINING')
pp.text(spdates[TRAINING+INPUT+50],numpy.max(spvalues)-50,'PREDICTION')
pp.grid(True)

pp.subplot(313)
pp.plot(numpy.reshape(errors[0],(len(errors[0]),1)))
pp.xlabel("Epoch Number")
pp.ylabel("Mean-Squared Error")
pp.grid(True)

pp.show()