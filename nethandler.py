from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import LinearLayer, FullConnection, FeedForwardNetwork, BiasUnit, TanhLayer
import pandas as pan


class NetHandler():
    net = ''
    series = pan.Series()
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
        n.addOutputModule(TanhLayer(self.OUTS, name="out"))

        n.addConnection(FullConnection(n['bias'], n['in']))
        n.addConnection(FullConnection(n['in'], n['h1']))
        n.addConnection(FullConnection(n['h1'], n['out']))
        n.sortModules()
        n = n.convertToFastNetwork()
        n.randomize()
        self.net = n

    def create_training_data(self, series, TRAINING):
        self.series = series
        self.data = SupervisedDataSet(self.INS, self.OUTS)
        for i in range(0, TRAINING - self.INS - self.OUTS):
            ins = series.values[i:i + self.INS]
            targets = series.values[i + self.INS:i + self.INS + self.OUTS]
            self.data.addSample(ins, targets)

    def train(self, LRATE, MOMENTUM, ITERATIONS):
        trainer = BackpropTrainer(self.net, self.data, learningrate=LRATE, momentum=MOMENTUM, weightdecay=0.0001)
        print "Training..."
        return trainer.trainUntilConvergence(maxEpochs=ITERATIONS)

    def get_output(self, TRAINING, TESTING):
        outputs = []
        for i in range(TRAINING, TRAINING + TESTING - self.INS - self.OUTS):
            outputs.extend(self.net.activate(self.series.values[i:i + self.INS]))
        return pan.Series(outputs, self.series.index[TRAINING + self.INS:TRAINING + TESTING - self.OUTS])

