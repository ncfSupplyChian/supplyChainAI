import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import app.algorithm.algorithm_util as au

# 用pandas加载数据.csv文件，然后用train_test_split分成训练集和测试集
data = pd.read_csv('E:\\code\\python\\data\\dshl_lr.csv')
feature_cols = ['最近登录间隔', '历史月均订单数量', '交易品牌数', '客单价', '注册时间长度',
                '近6个月月均交易笔数', '近6个月月均交易SKU数', '近6个月月均交易金额', '店商互联信用得分', '取消订单次数']
x = data[feature_cols]
y = data['是否有退货']

# 归一化
x = preprocessing.scale(x)
# x = preprocessing.MinMaxScaler().fit_transform(x)
# print(x[0])
# x = preprocessing.MaxAbsScaler().fit_transform(x)
# print(x[0])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.9)

# 用支持向量机分类
svm = SVC(probability=True)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)

au.validate_result(svm, X_train, y_train)

au.draw_roc(svm, X_test, y_test)

au.draw_ks(svm, X_test, y_test)
