import random


def prepare(pre_data, split):
    lines = []
    for data in pre_data:
        line = ""
        for attr in data:
            line += attr + split
        line = line[:len(line) - 1] + "\n"
        lines.append(line)
    return lines


file = input("Specify dataset file(supported: .csv,.data):")
try:
    data = open(file)
    percentage = input("Specify percentage for the test set\nan advised choice is 30: ")
    percentage = float(percentage)
    splitter = ","
    if file.endswith(".csv"):
        splitter = ";"
    data_list = [line.replace('\n', '').split(splitter) for line in data]
    test_data = []
    length = int(round((len(data_list) * percentage) / 100))
    for i in range(length):
        rand = random.randint(0, len(data_list) - 1)
        test_data.append(data_list.pop(rand))
    f_t = open(file[:file.rfind(".data")] + "_train.data", "w")
    for line in prepare(data_list, splitter):
        f_t.write(line)
    f_t.close()
    f_t = open(file[:file.rfind(".data")] + "_test.data", "w")
    for line in prepare(test_data, splitter):
        f_t.write(line)
    f_t.close()
except:
    exit(-1)
