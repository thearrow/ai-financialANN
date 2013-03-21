from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import LinearLayer, FullConnection, FeedForwardNetwork, BiasUnit, TanhLayer
import pandas as pan
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

    def __init__(self, INS, HIDDEN, OUT):
        self.INS = INS
        self.HIDDEN = HIDDEN
        self.OUTS = OUT
        self.assemble_network()

    def assemble_network(self):
        n = FeedForwardNetwork()
        n.addModule(BiasUnit(name="bias"))
        n.addInputModule(LinearLayer(self.INS, name="in"))
        n.addModule(TanhLayer(self.HIDDEN, name="h1"))
        n.addModule(TanhLayer(self.HIDDEN, name="h2"))
        n.addOutputModule(TanhLayer(self.OUTS, name="out"))

        n.addConnection(FullConnection(n['bias'], n['in']))
        n.addConnection(FullConnection(n['in'], n['h1']))
        n.addConnection(FullConnection(n['h1'], n['h2']))
        n.addConnection(FullConnection(n['h2'], n['out']))
        n.sortModules()
        n = n.convertToFastNetwork()
        n.randomize()
        self.net = n

    def create_training_data(self, handler, TRAINING):
        self.handler = handler
        self.indata = handler.data
        self.data = SupervisedDataSet(self.INS, self.OUTS)
        for i in range(0, (TRAINING - self.OUTS)):
            ins = self.indata.ix[i].values
            target = self.indata.ix[i + 1].values[0]
            self.data.addSample(ins, target)

    def train(self, LRATE, MOMENTUM, ITERATIONS):
        trainer = BackpropTrainer(self.net, self.data, learningrate=LRATE, momentum=MOMENTUM, weightdecay=0.0001)
        print "Training..."
        return trainer.trainUntilConvergence(maxEpochs=ITERATIONS)

    def get_output(self, TRAINING, TESTING):
        outputs = []
        end_index = TRAINING + TESTING
        for i in range(TRAINING, end_index):
            ins = self.indata.ix[i].values
            outputs.extend(self.net.activate(np.array(ins)))
        return pan.Series(outputs, self.indata.index[TRAINING:end_index])
