import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score


def get_data(test_size):
    data = pd.read_csv('E:\\code\\python\\data\\dshl_lr.csv')
    feature_cols = ['最近登录间隔', '历史月均订单数量', '交易品牌数', '客单价', '注册时间长度',
                    '近6个月月均交易笔数', '近6个月月均交易SKU数', '近6个月月均交易金额', '店商互联信用得分', '取消订单次数']
    x = data[feature_cols]
    y = data['是否有退货']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    return x_train, x_test, y_train, y_test


def validate_result(clf, x, y):
    # 效果评估：
    # 准确率：scikit-learn提供了accuracy_score来计算：LogisticRegression.score()
    # 准确率是分类器预测正确性的比例，但是并不能分辨出假阳性错误和假阴性错误
    scores = cross_val_score(clf, x, y, cv=5)
    print('准确率：', np.mean(scores), scores)

    # 精确率和召回率：
    # 精确率是指分类器预测出的有退货小B中真的是退货小B的比例，P=TP/(TP+FP)
    # 召回率在医学上也叫做灵敏度，在本例中知所有真的退货小B被分类器正确找出来的比例，R=TP/(TP+FN)
    precisions = cross_val_score(clf, x, y, cv=5, scoring='precision')
    print('精确率：', np.mean(precisions), precisions)
    recalls = cross_val_score(clf, x, y, cv=5, scoring='recall')
    print('召回率：', np.mean(recalls), recalls)
    # 综合评价指标（F1 measure）是精确率和召回率的调和均值（harmonic mean），或加权平均值，也称为F-measure或fF-score。
    f1s = cross_val_score(clf, x, y, cv=5, scoring='f1')
    print('综合评价指标：', np.mean(f1s), f1s)
    roc_auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc')
    print('roc_auc：', np.mean(roc_auc), roc_auc)


def draw_roc(clf, x, y):
    probas_ = clf.predict_proba(x)
    fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=1,
             label='ROC fold (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def draw_ks(clf, x, y):
    proba = clf.predict_proba(x)
    fpr, tpr, th = roc_curve(y, proba[:, 1])
    x_axis = np.arange(len(fpr)) / float(len(fpr))
    plt.figure(figsize=[6, 6])
    ks = max(abs(fpr - tpr))
    print('KS值：',ks)
    plt.plot(fpr, x_axis, color='blue', label='fpr')
    plt.plot(tpr, x_axis, color='red', label='tpr')
    plt.legend(loc="lower right")
    title = 'KS curve ks = ' + str(ks)
    plt.title(title)
    plt.show()


def validate_grid(pipeline, parameters, x_train, y_train, x_test, y_test):
    # 网格搜索（Grid search）就是用来确定最优超参数的方法。
    # 其原理就是选取可能的参数不断运行模型获取最佳效果。
    # 网格搜索用的是穷举法，其缺点在于即使每个超参数的取值范围都很小，
    # 计算量也是巨大的。不过这是一个并行问题，参数与参数彼此独立，
    # 计算过程不需要同步，所有很多方法都可以解决这个问题。scikit-learn有GridSearchCV()函数解决这个问题：
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='roc_auc', cv=3)
    grid_search.fit(x_train, y_train)
    print('最佳效果：%0.3f' % grid_search.best_score_)
    print('最优参数组合：')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))
    predictions = grid_search.predict(x_test)
    print('准确率：', accuracy_score(y_test, predictions))
    print('精确率：', precision_score(y_test, predictions))
    print('召回率：', recall_score(y_test, predictions))
    probas_ = grid_search.predict_proba(x_test)
    print('ROC_AUC：', roc_auc_score(y_test, probas_[:, 1]))
    print(classification_report(y_test, predictions))
