from math import floor
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import *
import pandas as pan
from pandas.tseries.offsets import *
import numpy as np


class NetHandler():
    net = ''
    indata = ''
    data = ''
    INS = 0
    HIDDEN = 0
    OUTS = 0
    initialization_periods = 50

    def __init__(self, INS, HIDDEN, OUT, data):
        self.INS = INS
        self.HIDDEN = HIDDEN
        self.OUTS = OUT
        self.indata = data.dataframe
        self.assemble_rn()
        #self.assemble_ffn()

    def assemble_rn(self):
        n = RecurrentNetwork()
        n.addInputModule(LinearLayer(self.INS, name="in"))
        n.addModule(LSTMLayer(self.HIDDEN, name="hidden"))
        n.addOutputModule(LinearLayer(self.OUTS, name="out"))
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

    def assemble_ffn(self):
        n = FeedForwardNetwork()
        n.addInputModule(LinearLayer(self.INS, name="in"))
        n.addModule(TanhLayer(self.HIDDEN, name="hidden"))
        n.addOutputModule(TanhLayer(self.OUTS, name="out"))
        n.addModule(BiasUnit(name="outbias"))
        n.addModule(BiasUnit(name="hidbias"))

        n.addConnection(FullConnection(n['in'], n['hidden']))
        n.addConnection(FullConnection(n['hidden'], n['out']))
        n.addConnection(FullConnection(n['hidbias'], n['hidden']))
        n.addConnection(FullConnection(n['outbias'], n['out']))
        n.sortModules()
        n.randomize()
        self.net = n

    def train(self, data, LRATE, MOMENTUM, ITERATIONS):
        trainer = BackpropTrainer(module=self.net, dataset=data, learningrate=LRATE,
                                  momentum=MOMENTUM, lrdecay=0.99999, verbose=True)
        for i in xrange(0, self.initialization_periods):
            self.net.activate(data.getSequence(i)[0])
        print "Training..."
        return trainer.trainUntilConvergence(maxEpochs=ITERATIONS)

    def get_output(self, data, proportion):
        outputs = []
        for i in range(0, len(data)):
            datum = data.getSequence(i)
            outs = self.net.activate(datum[0])
            outputs.extend(outs)
        index = self.indata.index[floor(len(self.indata) * proportion) + 1:]
        return pan.Series(outputs, index)

    def change_tomorrow(self):
        index = len(self.indata) - 1
        ins = self.indata.ix[index].values
        output = self.net.activate(np.array(ins))
        tomorchange = output[0]
        incdec = ""

        incdec += "\nOn %s the market will " % (self.indata.index[index] + BDay()).to_datetime().strftime("%a, %b %d, %Y")
        if tomorchange > 0:
            incdec += "increase.\n"
        else:
            incdec += "decrease.\n"

        return incdec

