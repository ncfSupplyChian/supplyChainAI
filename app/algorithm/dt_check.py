import pandas as pd
import numpy as np
from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# 用pandas加载数据.csv文件，然后用train_test_split分成训练集和测试集
data = pd.read_csv('E:\\code\\python\\data\\dshl_lr.csv')
feature_cols = ['最近登录间隔', '历史月均订单数量', '交易品牌数', '客单价', '注册时间长度',
                '近6个月月均交易笔数', '近6个月月均交易SKU数', '近6个月月均交易金额', '店商互联信用得分', '取消订单次数']
x = data[feature_cols]
y = data['是否有退货']

# 归一化
# x = preprocessing.scale(x)
# print(x[0])
# x = preprocessing.MinMaxScaler().fit_transform(x)
# print(x[0])
# x = preprocessing.MaxAbsScaler().fit_transform(x)
# print(x[0])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# 用决策树分类
dt = tree.DecisionTreeClassifier(max_depth=4)
# dt = RandomForestClassifier()
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
# with open("iris.dot", 'w') as f:
#     f = tree.export_graphviz(dt, out_file=f, feature_names=['recent_login', 'avg_monthly_order', 'trading_brands',
#                                                             'customer_price', 'reg_time', '6_months_avg_transactions',
#                                                             '6_months_avg_SKUs', '6_months_avg_transaction_amount',
#                                                             'dshl_score', 'cancel_order_number'])
print('测试集score：', dt.score(X_test, y_test))
print(dt.feature_importances_)
print(dt.classes_)
for i, prediction in enumerate(predictions[-1:]):
    print('预测类型：%s. 信息: %s' % (prediction, X_test.iloc[i]))

# 效果评估：
# 准确率：scikit-learn提供了accuracy_score来计算：LogisticRegression.score()
# 准确率是分类器预测正确性的比例，但是并不能分辨出假阳性错误和假阴性错误
scores = cross_val_score(dt, X_train, y_train, cv=5)
print('准确率：', np.mean(scores), scores)

# 精确率和召回率：
# 精确率是指分类器预测出的有退货小B中真的是退货小B的比例，P=TP/(TP+FP)
# 召回率在医学上也叫做灵敏度，在本例中知所有真的退货小B被分类器正确找出来的比例，R=TP/(TP+FN)
precisions = cross_val_score(dt, X_train, y_train, cv=5, scoring='precision')
print('精确率：', np.mean(precisions), precisions)
recalls = cross_val_score(dt, X_train, y_train, cv=5, scoring='recall')
print('召回率：', np.mean(recalls), recalls)
# 综合评价指标（F1 measure）是精确率和召回率的调和均值（harmonic mean），或加权平均值，也称为F-measure或fF-score。
f1s = cross_val_score(dt, X_train, y_train, cv=5, scoring='f1')
print('综合评价指标：', np.mean(f1s), f1s)
