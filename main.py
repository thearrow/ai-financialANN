import csv, numpy, os.path, itertools, ystockquote
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from matplotlib import pyplot
from matplotlib import dates

#Financial Data
sp500filename = 'sp500.csv'
startdate = '20020101' #YYYYMMDD
enddate = '20130220' #YYYYMMDD
sp500mean = 0
sp500max = 0

#Neural Network
INPUT = 20
HIDDEN = 15
OUTPUT = 1
ITERATIONS = 20
TRAINING = 2300
TESTING = 500
LRATE = 0.05

#fetch financial data from file or yahoo API
def load_index(file):
    if os.path.isfile(file):
        data = list(csv.reader(open(file,'rb'),delimiter=','))
        data.reverse()
        return data
    else:
        data = ystockquote.get_historical_prices("%5EGSPC",startdate,enddate)
        data.pop(0)
        with open(file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        data.reverse()
        return data

#data normalization functions {-1:1}
def normalize(data):
    mean = numpy.mean(list(float(d[-1]) for d in data))
    max = numpy.max(list(numpy.abs(float(d[-1])-mean) for d in data))
    global sp500mean, sp500max
    sp500mean = mean
    sp500max = max
    for i,val in enumerate(data):
        data[i][-1] = (float(val[-1])-mean)/max

def un_normalize(data):
    for i,val in enumerate(data):
        data[i] = (float(val)*sp500max)+sp500mean

#Neural Net Functions
def train(net, data):
    trainer = BackpropTrainer(net, data, learningrate=LRATE, momentum=0.9, weightdecay=0.0001)
    #for _ in range(ITERATIONS):
    #    print trainer.train()
    print "Training..."
    trainer.trainUntilConvergence(maxEpochs=ITERATIONS,verbose=True)

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

def get_output_dates(input):
    return list((dates.datestr2num(d[0]) for d in input[TRAINING+INPUT:TRAINING+TESTING-OUTPUT]))



#Main program

sp500 = load_index(sp500filename)
normalize(sp500)
spdates = list((dates.datestr2num(s[0]) for s in sp500))
spvalues = list((s[-1] for s in sp500))

net = buildNetwork(INPUT,HIDDEN,OUTPUT)
net.randomize()

data = create_training_data(spvalues)
train(net,data)

sp_predicted = get_output_vals(net, spvalues)
un_normalize(spvalues)
un_normalize(sp_predicted)

pyplot.plot_date(spdates,spvalues,linestyle='solid',c='b',marker='None')
pyplot.plot_date(get_output_dates(sp500),sp_predicted,linestyle='solid',c='r',marker='None')
pyplot.vlines(spdates[TRAINING+INPUT],1000,1600)

pyplot.show()