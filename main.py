from ada_core import AdaBoost
from dataset import try_read


# AdaBoost
def ada_init(dataset):
    while True:
        iterations = input("Specify iteration number. A value greater than 100 is advised:\n")
        if any(l.isalpha() for l in iterations):
            print("Your input contained one or more letters, please retry\n")
        else:
            print("Boosted learning initiated.")
            return AdaBoost(data_set, int(iterations))


# Main sequence
# Print instructions and import the dataset
print(open("information.info").read())
file = input("Input DataSet path: ")
data_set = try_read(file)
print("Training dataset: " + file + " imported\n")
# Ask the user about the train/test distinction
case = 0
while True:
    ch = input("Would you like to split the dataset for training and testing?(Y/N)")
    if ch == 'N' or ch == 'n':
        case = 1
        break
    elif ch == 'Y' or ch == 'y':
        case = 2
        data_set.split()
        break
    else:
        print("wrong input retry")

# Test set choice according to the user's answer
if case == 1:
    test_set = try_read(input("Input the name of the desired test set: "))
    examples = test_set.get_examples()
    answers = test_set.Y
else:
    examples = data_set.trainX
    answers = data_set.trainY

# Algorithm choice
learner = ada_init(data_set)
right = 0
# Do the test
for example in examples:
    pred = learner.predict(example)
    if pred == answers[examples.index(example)]:
        right += 1
print("Right: " + str(right) + " out of: " + str(len(examples)))
file = open("results.txt", "a")
file.write("Training in M = " + str(learner.reps) + ":\n")
for line in learner.error_stat:
    file.write(str(line) + "\n")
file.write("Testing Success Rare\n")
file.write(str(right / len(examples)) + "\n")
file.close()
while True:
    right = 0
    ret = input("retrain ?(1-0)")
    if int(ret) == 1:
        learner = ada_init(data_set)
        for example in examples:
            pred = learner.predict(example)
            if pred == answers[examples.index(example)]:
                right += 1
    else:
        break
    print("Right: " + str(right) + " out of: " + str(len(examples)))
