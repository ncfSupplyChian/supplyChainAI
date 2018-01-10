from sklearn.ensemble import GradientBoostingClassifier
import app.algorithm.algorithm_util as au

x_train, x_test, y_train, y_test = au.get_data(0.5)

# 用GBDT分类
gbdt = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120, max_depth=3,
                                  min_samples_leaf=90, min_samples_split=700, subsample=0.85)
gbdt.fit(x_train, y_train)
predictions = gbdt.predict(x_test)

print(gbdt.feature_importances_)
for i, prediction in enumerate(predictions[-1:]):
    print('预测类型：%s. 信息: %s' % (prediction, x_test.iloc[i]))

au.validate_result(gbdt, x_test, y_test)

au.draw_roc(gbdt, x_test, y_test)

au.draw_ks(gbdt, x_test, y_test)
