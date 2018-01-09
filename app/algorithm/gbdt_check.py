import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

# 用GBDT分类
gbdt = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120, max_depth=3,
                                           min_samples_leaf=90, min_samples_split=700, subsample=0.85)
gbdt.fit(X_train, y_train)
predictions = gbdt.predict(X_test)

print('测试集score：', gbdt.score(X_test, y_test))
print(gbdt.feature_importances_)
for i, prediction in enumerate(predictions[-1:]):
    print('预测类型：%s. 信息: %s' % (prediction, X_test.iloc[i]))

# 效果评估：
# 准确率：scikit-learn提供了accuracy_score来计算：LogisticRegression.score()
# 准确率是分类器预测正确性的比例，但是并不能分辨出假阳性错误和假阴性错误
scores = cross_val_score(gbdt, X_train, y_train, cv=5)
print('准确率：', np.mean(scores), scores)

# 精确率和召回率：
# 精确率是指分类器预测出的有退货小B中真的是退货小B的比例，P=TP/(TP+FP)
# 召回率在医学上也叫做灵敏度，在本例中知所有真的退货小B被分类器正确找出来的比例，R=TP/(TP+FN)
precisions = cross_val_score(gbdt, X_test, y_test, cv=5, scoring='precision')
print('精确率：', np.mean(precisions), precisions)
recalls = cross_val_score(gbdt, X_test, y_test, cv=5, scoring='recall')
print('召回率：', np.mean(recalls), recalls)
# 综合评价指标（F1 measure）是精确率和召回率的调和均值（harmonic mean），或加权平均值，也称为F-measure或fF-score。
f1s = cross_val_score(gbdt, X_test, y_test, cv=5, scoring='f1')
print('综合评价指标：', np.mean(f1s), f1s)
roc_auc = cross_val_score(gbdt, X_test, y_test, cv=5, scoring='roc_auc')
print('roc_auc：', np.mean(roc_auc), roc_auc)

# ROC曲线可以用来可视化分类器的效果。和准确率不同，ROC曲线对分类比例不平衡的数据集不敏感，
# ROC曲线显示的是对超过限定阈值的所有预测结果的分类器效果。
# ROC曲线画的是分类器的召回率与误警率（fall - out）的曲线。误警率也称假阳性率，
# 是所有阴性样本中分类器识别为阳性的样本所占比例：F = FPTN + FP
# AUC是ROC曲线下方的面积，它把ROC曲线变成一个值，表示分类器随机预测的效果。scikit - learn提供了计算ROC和AUC指标的函数
# Compute ROC curve and area the curve
probas_ = gbdt.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, lw=1, alpha=0.3,
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