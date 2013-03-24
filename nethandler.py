from pybrain.datasets import SequentialDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import *
import pandas as pan
from pandas.tseries.offsets import *
import datahandler as dh
import numpy as np


class NetHandler():
    net = ''
    handler = dh.DataHandler()
    indata = ''
    data = ''
    INS = 0
    HIDDEN = 0
    OUTS = 0
    initialization_periods = 50

    def __init__(self, INS, HIDDEN, OUT):
        self.INS = INS
        self.HIDDEN = HIDDEN
        self.OUTS = OUT
        self.assemble_network()

    def assemble_network(self):
        n = RecurrentNetwork()
        n.addInputModule(LinearLayer(self.INS, name="in"))
        n.addModule(LSTMLayer(self.HIDDEN, name="hidden"))
        n.addOutputModule(TanhLayer(self.OUTS, name="out"))
        n.addModule(BiasUnit(name="outbias"))
        n.addModule(BiasUnit(name="hidbias"))

        n.addConnection(FullConnection(n['in'], n['hidden']))
        n.addRecurrentConnection(FullConnection(n['hidden'], n['hidden']))
        n.addConnection(FullConnection(n['hidden'], n['out']))
        n.addConnection(FullConnection(n['hidbias'], n['hidden']))
        n.addConnection(FullConnection(n['outbias'], n['out']))
        n.sortModules()
        n.randomize()
        self.net = n

    def create_training_data(self, handler, TRAINING):
        self.handler = handler
        self.indata = handler.data
        self.data = SequentialDataSet(self.INS, self.OUTS)
        for i in xrange(self.initialization_periods, TRAINING):
            self.data.newSequence()
            ins = self.indata.ix[i].values
            target = self.indata.ix[i + 1].values[0]
            self.data.appendLinked(ins, target)

    def train(self, LRATE, MOMENTUM, ITERATIONS):
        trainer = BackpropTrainer(module=self.net, dataset=self.data, learningrate=LRATE,
                                  momentum=MOMENTUM, lrdecay=0.9999, verbose=True)
        for i in xrange(0, self.initialization_periods):
            self.net.activate(self.indata.ix[i].values)
        print "Training..."
        # for _ in xrange(ITERATIONS):
        #     trainer.train()
        return trainer.trainUntilConvergence(maxEpochs=ITERATIONS)

    def get_output(self, TRAINING, TESTING):
        outputs = []
        start_index = TRAINING
        end_index = TRAINING + TESTING
        for i in xrange(start_index, end_index):
            ins = self.indata.ix[i].values
            outs = self.net.activate(np.array(ins))
            outputs.extend(outs)
        index = self.indata.index[start_index:end_index] + (self.OUTS * BDay())
        return pan.Series(outputs, index)

    def change_tomorrow(self):
        index = len(self.indata) - 1
        ins = self.indata.ix[index].values
        output = self.net.activate(np.array(ins))
        tomorchange = output[0]
        incdec = ""

        incdec += "On %s the market will " % (self.indata.index[index] + BDay()).to_datetime().strftime("%a, %b %d, %Y")
        if tomorchange > 0:
            incdec += "increase."
        else:
            incdec += "decrease."

        return incdec

