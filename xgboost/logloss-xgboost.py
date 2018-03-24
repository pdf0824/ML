# coding=utf-8
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
dataset = loadtxt('./pima-indians-diabetes.csv', delimiter=',')

# 取特征值和label值
X = dataset[:, 0:8]
Y = dataset[:, 8]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier()
eval_set = [(X_test, y_test)]
'''
early_stopping_rounds:如果连续N次性能没有提升，则不再训练
eval_metric：指定用什么loss作为评估标准
eval_set：新加入一个模型，使用eval_set测试
verbose：显示信息
'''
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='logloss', eval_set=eval_set, verbose=True)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print('Accuracy:%.2f%%' % (accuracy * 100))
