# coding=utf-8
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

# 加载数据
dataset = loadtxt('./pima-indians-diabetes.csv', delimiter=',')

# 取特征值和label值
X = dataset[:, 0:8]
Y = dataset[:, 8]

model = XGBClassifier()
model.fit(X, Y)

plot_importance(model)
pyplot.show()
