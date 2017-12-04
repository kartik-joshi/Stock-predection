from csv import reader
from sklearn import svm
import numpy as np
import time

# Read datasets from CSV input file
def Read_file(file_name):
    dataset = list()
    with open(file_name, 'r', newline='',encoding='utf-8') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string columns to float in input dataset
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer in input dataset (last column with class value)
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


start_time = time.time()

# testdataset
testdataset = Read_file('SVM_Input.csv')
test_target= Read_file('SVM_Target.csv')

n_epoch = 30
w = 0
r = 1
runs = [0]*30

for j in range(len(runs)):
    # # learning_rate = learning_rate + 0.05
    # # n_epoch = n_epoch + 4
    r =  r + 1000
    if (r + 10000) > 15800:
        r = w
        w += 300
    traindataset = [testdataset[i] for i in range(r, r + 10000)]
    train_target = [test_target[i] for i in range(r, r + 10000)]
    clf = svm.SVC(kernel="rbf", decision_function_shape='ovo')
    clf.fit(traindataset, train_target)

    # test network
    count = 0
    total = 0
    for i in range(0, len(testdataset)):
        total += 1
        temp = clf.predict([testdataset[i]])
        if temp == test_target[i]:
            count += 1
    accuracy = count * 100 / total
    print('Accuracy: %s' % accuracy)
    runs[j] = accuracy

mean = sum(runs)/len(runs)
print("n_epoch: {}".format(n_epoch))
print("Mean_Accuracy: {}".format(mean))
print("Standard_Deviation: {}".format(np.std(runs, 0)))
time_taken = time.time() - start_time
print("total_time : {}".format(time_taken))

