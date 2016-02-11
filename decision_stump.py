from copy import deepcopy
from math import inf
from dataset import DataSet


class DecisionStumpLearner:
    class Operation:
        # Made this way for better reading
        # Thus you init like Operation(5, <=,<, 10)
        def __init__(self, floor, op1, op2, ceiling):
            self.ceiling = ceiling
            self.floor = floor
            self.op1 = op1
            self.op2 = op2

        def val_check(self, val):
            res = 2
            if self.op1 is None:
                res -= 1
            elif self.op1 == "<=" and self.floor <= float(val):
                res -= 1
            elif self.op1 == "<" and self.floor < float(val):
                res -= 1
            elif self.op1 == "=" and self.floor == val:
                res -= 1
            if self.op2 is None:
                res -= 1
            elif self.op2 == "<=" and float(val) <= self.ceiling:
                res -= 1
            elif self.op2 == "<" and float(val) < self.ceiling:
                res -= 1
            elif self.op2 == "=" and self.ceiling == val:
                res -= 1
            return res == 0

    class DecisionStump:
        # init the stump using a name that is the index of the attribute or an operation object
        def __init__(self, name, value, operation):
            self.name = name
            self.val = value
            self.leafs = []
            self.operation = operation

        # Add a leaf to the stump using an operation and a tree
        def add_leaf(self, op, tree):
            if tree.name is None:
                tree.name = self.name
            tree.operation = op
            self.leafs.append(tree)

        # Predict the result of the example
        def predict(self, example):
            for leaf in self.leafs:
                if leaf.operation.val_check(example[int(self.name)]):
                    return leaf.val
            return None

    # Initiate the Learner, by copying the dataset, assigning weights, and triggering the learning process
    def __init__(self, data, weights, mode=0):
        self.data = deepcopy(data)
        self.w = weights
        if weights is not None:
            self.data.w = self.w
        self.data.repopulate()
        self.tree = self.train(self.data.get_examples(), self.data.attr_names, mode)

    # The training method
    def train(self, examples, attributes, default=None):
        if len(examples) == 0:
            return default
        # If all the example have the same result
        elif self.same_class(examples):
            return self.DecisionStump(None, self.data.Y[self.data.get_index(examples[0])], None)
        # If the attributes are empty
        elif attributes == 0:
            return self.DecisionStump(None, self.majority(examples), None)
        else:
            # Choose an attribute and return its index, split datasets and operations
            attr, splits, ops = self.choose_attr()
            # Create the stump for the attribute
            tree = self.DecisionStump(attr, None, None)
            # The default child is the majority of the results for the given examples
            default_new = self.DecisionStump(None, self.majority(examples), None)
            i = 0
            # for every data set that we have as output from the choose attr method
            for data in splits:
                # Get the examples contained in the said data set
                examples_n = data.get_examples()
                tree.add_leaf(ops[i], self.train(examples_n, 0, default_new))
                i += 1
            return tree

    # Predict the class of a given example
    def predict(self, example):
        return self.tree.predict(example)

    # Check if the prediction was correct given an example and a result
    def check_predict(self, example, target):
        prediction = self.predict(example)
        return prediction == target

    # Choose the attribute that's best to split by
    def choose_attr(self):
        max_gain = -inf
        name = 0
        max_splits = []
        max_ops = None
        for attr in range(len(self.data.X[0])):
            gain, splits, ops = self.get_gain(attr)
            if gain > max_gain:
                max_gain = gain
                max_splits = splits
                max_ops = ops
                name = attr
        return name, max_splits, max_ops

    # Get the gain for an attribute
    def get_gain(self, attr):
        temp, ops = self.split_by(attr)
        size = self.data.get_size_weighted()
        remainder = 0.0
        for val in temp:
            new_size = val.get_size_weighted()
            if size == 0:
                remainder = 0
            else:
                remainder += (new_size / size) * val.calc()
        return self.data.calc() - remainder, temp, ops

    # Try to split the dataset by an attribute
    def split_by(self, attr):
        res = {}
        examples = self.data.get_examples()
        vals = self.data.get_values()
        if len(vals[attr]) > 10 and not any(l.isalpha() for l in vals[attr][0]):
            return self.enhanced_split(attr, vals[attr])
        for i in range(len(examples)):
            val = examples[i][attr]
            if val not in res:
                res[val] = DataSet(None)
            res[val].X.append(examples[i])
            res[val].Y.append(self.data.Y[i])
            res[val].times.append(self.data.times[i])
        ops = []
        results = []
        for key in res:
            ops.append(self.Operation(key, "=", None, None))
            results.append(res[key])
        return results, ops

    # Enhanced split used for continuous values
    def enhanced_split(self, attr, vals):
        maxi = mini = float(vals[0])
        res = []
        for val in vals:
            if float(val) > maxi:
                maxi = float(val)
            if float(val) < mini:
                mini = float(val)
        times = len(self.data.get_results())
        for i in range(times):
            res.append(DataSet(None))
        floor = mini
        step = (maxi - mini) / times
        examples = self.data.get_examples()
        ops = []
        for i in range(len(examples)):
            for j in range(times):
                if j == times - 1:
                    if (floor + (step * j)) <= float(examples[i][attr]):
                        res[j].X.append(examples[i])
                        res[j].Y.append(self.data.Y[i])
                        res[j].times.append(self.data.times[i])
                        if len(ops) - 1 != j:
                            ops.append(self.Operation((floor + (step * j)), "<=", None, None))
                        break
                if (floor + (step * j)) <= float(examples[i][attr]) < ((floor + step) + (step * j)):
                    res[j].X.append(examples[i])
                    res[j].Y.append(self.data.Y[i])
                    res[j].times.append(self.data.times[i])
                    if len(ops) - 1 != j:
                        ops.append(self.Operation((floor + (step * j)), "<=", "<", ((floor + step) + (step * j))))
                    break
        return res, ops

    # Check if the examples have the same classes
    def same_class(self, examples):
        org = self.data.get_result(examples[0])
        for example in examples[1:]:
            if self.data.get_result(example) != org:
                return False
        return True

    # For a given list of examples find the most common class
    def majority(self, examples):
        vals = {}
        for example in examples:
            index = self.data.get_index(example)
            res = self.data.Y[index]
            if res not in vals:
                vals[res] = self.data.times[index]
            else:
                vals[res] += self.data.times[index]
        maxed = -5
        val = None
        for key in vals.keys():
            if vals[key] > maxed:
                maxed = vals[key]
                val = key
        return val
