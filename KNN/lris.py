# coding=utf-8
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()

# print(iris)
knn.fit(iris.data, iris.target)
predictedLabel = knn.predict([[5.3, 0.4, 1.3, 1.4]])
print(predictedLabel)
