import random
from copy import deepcopy
from math import log


def try_read(file):
    while True:
        raw = None
        try:
            raw = open(file)
        finally:
            if raw is None:
                return None
            else:
                splitter = ''
                if file.endswith(".csv"):
                    splitter = ';'
                elif file.endswith(".data"):
                    splitter = ','

                data_t = [line.replace('\n', '').split(splitter) for line in raw]

                if data_t is not None:
                    return DataSet(data_t)

                else:
                    print("The specified path does not contain a readable file or\n "
                          "the file is not of supported format, please retry\n")


class DataSet:
    """
This class should be able to read pretty much any given data set and provide examples in the form
of lists, or something else still figuring it out
Thought: It is advised to use a custom iterator
"""

    def __init__(self, file_data):
        self.X = []
        self.Y = []
        self.w = []
        self.trainX = []
        self.trainY = []
        self.times = []
        if file_data is not None:
            self.values = []
            self.results = []
            self.raw_data = file_data
            self.consume()
            self.attr_names = ["Attribute_" + str(i) for i in range(len(self.X[0]))]
            self.val_discover()
            self.times = [1 for ex in self.X]

    # Consume the data from the file
    def consume(self):
        for data in self.raw_data:
            self.X.append(data[:len(data) - 1])
            self.Y.append(data[len(data) - 1])

    # Discover all the possible values for each example
    def val_discover(self):
        for i in range(len(self.X[0])):
            self.values.append([])
        for example in self.X:
            i = 0
            for value in example:
                if not self.values[i].__contains__(value):
                    self.values[i].append(value)
                i += 1
        self.results.extend(x for x in self.Y
                            if not self.results.__contains__(x))

    # self explanatory, you never know when it's going to be needed
    def get_examples(self):
        return deepcopy(self.X)

    # same
    def get_answers(self):
        return self.Y

    # get the possible values for each attribute
    def get_values(self):
        return self.values

    # get the possible results for any example that matches this set
    def get_results(self):
        return self.results

    # get the data of a particular example
    def get(self, index):
        return self.X[index], self.Y[index]

    # get an example by index
    def get_index(self, example):
        return self.X.index(example)

    def get_result(self, example):
        return self.Y[self.X.index(example)]

    # return the size of the DataSet
    def ex_size(self):
        return len(self.Y)

    # Return the example whose desired attribute, defined by index, is equal to val
    def get_matching(self, index, val):
        return [example for example in self.X
                if example[index] == val]

    def split(self):
        length = int(len(self.X) / 3)
        for i in range(length):
            rand = random.randint(0, len(self.X) - 1)
            self.trainX.append(self.X.pop(rand))
            self.trainY.append(self.Y.pop(rand))

    def get_size_weighted(self):
        size = 0
        for time in self.times:
            size += time
        return size

    def repopulate(self):
        length = len(self.X)
        for i in range(length):
            first = int(self.w[i] * length)
            second = (self.w[i] * length) % 1
            if second != 0:
                second = 1
            if first == 0:
                second = 0
            self.times[i] = first + second

    # Calculate the entropy of this data set
    def calc(self):
        res = {}
        i = 0
        for ex in self.Y:
            if ex not in res:
                res[ex] = self.times[i]
            else:
                res[ex] += self.times[i]
            i += 1

        sume = sum(self.times)
        if sume == 0:
            return 0
        norm = [val / sume for val in res.values()]
        total = 0.0
        for nor in norm:
            if nor == 0:
                continue
            else:
                total += (-1) * (nor * log(nor))
        return total
