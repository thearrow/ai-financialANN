import itertools
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer


class TrainerHandler():

    def __init__(self, i, cd, ins, outs, it, train, test, lr, mom):
        self.indices = i
        self.change_days = cd
        self.ins = ins
        self.outs = outs
        self.iterations = it
        self.train = train
        self.test = test
        self.learnrate = lr
        self.momentum = mom
        self.data = ''
        self.trainer = ''

    def create_data(self, datahandlers, target_index):
        self.data = SupervisedDataSet(self.ins, self.outs)
        count = 0
        cds = self.change_days
        while count + cds + self.outs < self.train:
            ins = self.input_slice(count, cds, datahandlers)
            out_changes = datahandlers[target_index].get_changes()
            outs = out_changes[count + cds + 1:count + cds + 1 + self.outs][0][0]
            self.data.addSample(ins, outs)
            count += 1

    def input_slice(self, i, days, datahandlers):
        ins = []
        for dh in datahandlers:
            changes = dh.get_changes()
            lol = changes[i + days - 1:i + days]
            l = [val for subl in lol for val in subl]
            ins.extend(l)
        return ins

    def perform_training(self, net):
        t = BackpropTrainer(net, self.data, learningrate=self.learnrate, momentum=self.momentum, weightdecay=0.0001)
        print "Training..."
        return t.trainUntilConvergence(maxEpochs=self.iterations)

    def get_output_vals(self, net, datahandlers):
        outputs = []
        days = self.change_days
        count = self.train
        while count + days + self.outs < self.train + self.test:
            ins = self.input_slice(count, days, datahandlers)
            outputs.append(net.activate(ins))
            count += self.outs
        realouts = []
        for d in outputs:
            realouts.append(list(d))
        return list(itertools.chain(*realouts))

    def get_output_dates(self, dates):
        return dates[self.train + self.change_days:self.train + self.test - self.outs]