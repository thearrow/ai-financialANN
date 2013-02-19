from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from matplotlib import pyplot
from matplotlib import dates
import numpy,sys

array = numpy.loadtxt('SP500-3yr.csv',delimiter=',',converters = {0: dates.datestr2num})
INPUT = 12
HIDDEN = 4
OUTPUT = 1
ITERATIONS = 100
TRAINING = 550
PREDICT = len(array)

data = SupervisedDataSet(INPUT,OUTPUT)

#normalize data
mean = numpy.mean(array[:,-1])
max = numpy.max(array[:,-1]-mean)
def normalize(arr):
    for i,val in enumerate(arr):
        arr[i] = (val-mean)/max

def unNormalize(arr):
    for i,val in enumerate(arr):
        arr[i] = (val*max)+mean

normalize(array[:,-1])


def train(net):
    if len(inputs) != INPUT:
        print "ERROR! #of inputs != DAYS"

    trainer = BackpropTrainer(net, data, learningrate=0.1, lrdecay=0.9999, momentum=0.85, weightdecay=0.0001)
    for _ in range(ITERATIONS):
        print trainer.train()

if sys.argv[1] == 'test':
    print 'Running...'

    #hiddenclass=LSTMLayer, outclass=SigmoidLayer
    net = buildNetwork(INPUT,HIDDEN,OUTPUT, hiddenclass=TanhLayer)
    net.randomize()

    #TRAIN
    count = 0
    while count+INPUT < TRAINING:
        inputs = list(array[count:count+INPUT,-1])
        output = array[count:count+INPUT+1,-1][-1]
        data.addSample(inputs, output)
        count += 1

    train(net)

    #PREDICT
    outputs = []
    count = TRAINING
    while count+INPUT < PREDICT:
        inputs = list(array[count:count+INPUT,-1])
        outputs.append(net.activate(inputs)[0])
        count += 1

    datenums = list(array[TRAINING+INPUT:PREDICT,0])

    unNormalize(array[:,-1])
    unNormalize(outputs)

    NetworkWriter.writeToFile(net, 'sp500.xml')
    print 'Network saved to sp500.xml'

    pyplot.plot_date(array[:,0],array[:,-1],linestyle='solid',c='b',marker='None')
    pyplot.plot_date(datenums,outputs,linestyle='solid',c='r',marker='None')
    pyplot.vlines(array[:,0][TRAINING+INPUT],1000,1600)

    pyplot.show()


