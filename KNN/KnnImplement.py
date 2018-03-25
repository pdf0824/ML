# coding=utf-8
import csv
import random
import math
import operator


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        datasets = list(lines)
        for x in range(len(datasets) - 1):
            for y in range(4):
                datasets[x][y] = float(datasets[x][y])
                if random.random() < split:
                    trainingSet.append(datasets[x])
                else:
                    testSet.append(datasets[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distance = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distance.append((trainingSet[x], dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVote = sorted(classVotes, key=operator.itemgetter(1), reverse=True)
    return sortedVote[0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset(r'./iris.data.txt', split, trainingSet, testSet)
    print('train set:' + repr(len(trainingSet)))
    print('test set:' + repr(len(testSet)))
    predictions = []
    k = 4
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('>predicted = ' + repr(result) + ', actual = ' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy:' + repr(accuracy) + '%')


main()
