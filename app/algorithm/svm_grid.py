from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import app.algorithm.algorithm_util as au
from sklearn import preprocessing
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('E:\\code\\python\\data\\dshl_lr.csv')
    feature_cols = ['最近登录间隔', '历史月均订单数量', '交易品牌数', '客单价', '注册时间长度',
                    '近6个月月均交易笔数', '近6个月月均交易SKU数', '近6个月月均交易金额', '店商互联信用得分', '取消订单次数']
    x = data[feature_cols]
    y = data['是否有退货']
    x = preprocessing.scale(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9)
    pipeline = Pipeline([
        ('clf', SVC(probability=True))
    ])
    parameters = {
        'clf__C': (0.1, 1, 10),
        # 'clf__gamma': ('auto', 1, 0.1, 0.01),
        # 'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        # 'clf__kernel': ('rbf', 'sigmoid'),
    }
    au.validate_grid(pipeline, parameters, x_train, y_train, x_test, y_test)
