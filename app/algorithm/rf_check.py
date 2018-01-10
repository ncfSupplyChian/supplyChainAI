from sklearn.ensemble import RandomForestClassifier
import app.algorithm.algorithm_util as au

x_train, x_test, y_train, y_test = au.get_data(0.5)

# 用决策树分类  criterion='entropy'
rf = RandomForestClassifier(max_depth=9, n_estimators=100)
rf.fit(x_train, y_train)
predictions = rf.predict(x_test)

print(rf.feature_importances_)
print(rf.classes_)
for i, prediction in enumerate(predictions[-1:]):
    print('预测类型：%s. 信息: %s' % (prediction, x_test.iloc[i]))

au.validate_result(rf, x_test, y_test)

au.draw_roc(rf, x_test, y_test)

au.draw_ks(rf, x_test, y_test)
