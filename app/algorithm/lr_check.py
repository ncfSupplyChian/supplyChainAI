from sklearn.linear_model.logistic import LogisticRegression
import app.algorithm.algorithm_util as au

x_train, x_test, y_train, y_test = au.get_data(0.5)

# 用逻辑回归分类
lr = LogisticRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)
print('β：', lr.coef_.tolist())
print('常量：', lr.intercept_.tolist())

au.validate_result(lr, x_test, y_test)

au.draw_roc(lr, x_test, y_test)

au.draw_ks(lr, x_test, y_test)
