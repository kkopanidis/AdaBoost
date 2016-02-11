from copy import deepcopy
from math import log
from decision_stump import DecisionStumpLearner


# By dividing the weights by the total sum we maintain the "hierarchy" of the weights while ensuring that sum < 1
def normalize(weight):
    total = sum(weight)
    if total != 0:
        return [w / total for w in weight]
    return weight


# Initialize the weights so tha they are the same and with sum 1
def weight_init(size):
    weights = []
    for i in range(size):
        weights.append(1. / size)
    return weights


class AdaBoost:
    # Train the algorithm
    def __init__(self, data_set, reps):
        self.reps = reps
        self.h = []
        self.z = []
        self.w = []
        self.data = data_set
        self.error_stat = []
        self.ada_boost(self.data, reps)

    # The main training method
    def ada_boost(self, data, rep):
        size = data.ex_size()
        ep = 1. / (2 * size)
        w = weight_init(size)
        for m in range(rep):
            learner = DecisionStumpLearner(data, w)
            self.h.append(deepcopy(learner))
            error = 0
            for j in range(size):
                ex, target = data.get(j)
                if not learner.check_predict(ex, target):
                    error = error + w[j]
            # Used to make sure that the error won't "over" or "under" flow
            # error = max(ep, min(error, 1. - ep))
            self.error_stat.append(deepcopy(error))
            for j in range(size):
                ex, target = data.get(j)
                if learner.check_predict(ex, target):
                    w[j] *= error / (1. - error)
            w = normalize(w)
            self.z.append(log((1 - error) / error))

    # Predict the result for the example
    def predict(self, example):
        self.mat = {}
        return self.weighted_majority(example)

    # Return the result with the most votes
    def weighted_majority(self, example):
        mat = self.predictions(example)
        score = -500
        val_name = ""
        for key in mat.keys():
            if mat[key] > score:
                score = mat[key]
                val_name = key
        return val_name

    # Gather the predictions of the learners
    def predictions(self, example):
        mat = {}
        for i in range(len(self.h)):
            val = self.h[i].predict(example)
            if val not in mat:
                mat[val] = self.z[i]
            else:
                mat[val] += self.z[i]
        return mat
