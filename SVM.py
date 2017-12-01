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



w = 0
r = 1

traindataset = testdataset[0:10000]
train_target = test_target[0:10000]
# clf = svm.SVC(decision_function_shape='ovo')
clf = svm.SVC(C=1.3,decision_function_shape='ovo')
clf.fit(traindataset, train_target)

#test network
count = 0
total  = 0
for i in range(10000,len(testdataset)):
    total +=1
    temp = clf.predict([testdataset[i]])
    print(str(temp)+"predected"+str(test_target[i])+"expected")
    if temp == test_target[i]:
        count+=1
accuracy = count*100/total
print('Accuracy: %s' % accuracy)







