from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import LinearLayer, FullConnection, FeedForwardNetwork, BiasUnit, TanhLayer
import pandas as pan
import datahandler as dh


class NetHandler():
    net = ''
    handler = dh.DataHandler()
    data = ''
    INS = 0
    HIDDEN = 0
    OUTS = 0
    value_series = pan.Series()

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
        n.addOutputModule(TanhLayer(self.OUTS, name="out"))

        n.addConnection(FullConnection(n['bias'], n['in']))
        n.addConnection(FullConnection(n['in'], n['h1']))
        n.addConnection(FullConnection(n['h1'], n['out']))
        n.sortModules()
        n = n.convertToFastNetwork()
        n.randomize()
        self.net = n

    def create_training_data(self, handler, TRAINING):
        self.handler = handler
        self.value_series = handler.value_series()
        self.data = SupervisedDataSet(self.INS, self.OUTS)
        for i in range(self.INS, TRAINING - self.OUTS):
            ins = self.get_inputs(i)
            target = self.handler.change_series(1)[i + self.OUTS]
            self.data.addSample(ins, target)

    def train(self, LRATE, MOMENTUM, ITERATIONS):
        trainer = BackpropTrainer(self.net, self.data, learningrate=LRATE, momentum=MOMENTUM, weightdecay=0.0001)
        print "Training..."
        return trainer.trainUntilConvergence(maxEpochs=ITERATIONS)

    def get_output(self, TRAINING, TESTING):
        outputs = []
        for i in range(TRAINING, TRAINING + TESTING - self.INS - self.OUTS):
            ins = self.get_inputs(i)
            outputs.extend(self.net.activate(ins))
        return pan.Series(outputs, self.value_series.index[TRAINING+2:TRAINING + TESTING - self.INS - self.OUTS+2])

    def get_inputs(self, i):
        ins = []
        for j in range(1, (self.INS / 2) + 1):
            ins.append(self.handler.change_series(j)[i])
        ins.extend(self.value_series[i - (self.INS / 2):i].values)
        return ins
