import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, roc_auc_score

if __name__ == '__main__':
    # 用pandas加载数据.csv文件，然后用train_test_split分成训练集和测试集
    data = pd.read_csv('E:\\code\\python\\data\\dshl_lr.csv')
    feature_cols = ['最近登录间隔', '历史月均订单数量', '交易品牌数', '客单价', '注册时间长度',
                    '近6个月月均交易笔数', '近6个月月均交易SKU数', '近6个月月均交易金额', '店商互联信用得分', '取消订单次数']
    x = data[feature_cols]
    y = data['是否有退货']
    x = preprocessing.scale(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.9)

    # 网格搜索（Grid search）就是用来确定最优超参数的方法。
    # 其原理就是选取可能的参数不断运行模型获取最佳效果。
    # 网格搜索用的是穷举法，其缺点在于即使每个超参数的取值范围都很小，
    # 计算量也是巨大的。不过这是一个并行问题，参数与参数彼此独立，
    # 计算过程不需要同步，所有很多方法都可以解决这个问题。scikit-learn有GridSearchCV()函数解决这个问题：
    pipeline = Pipeline([
        ('clf', SVC())
    ])
    parameters = {
        'clf__C': (0.1, 1, 10),
        'clf__gamma': ('auto', 1, 0.1, 0.01),
        # 'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        'clf__kernel': ('rbf', 'sigmoid'),
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    print(grid_search.cv_results_)
    print('最佳效果：%0.3f' % grid_search.best_score_)
    print('最优参数组合：')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))
    predictions = grid_search.predict(X_test)
    print('准确率：', accuracy_score(y_test, predictions))
    print('精确率：', precision_score(y_test, predictions))
    print('召回率：', recall_score(y_test, predictions))
    probas_ = grid_search.predict_proba(X_test)
    print('ROC_AUC：', roc_auc_score(y_test, probas_))
    print(classification_report(y_test, predictions))