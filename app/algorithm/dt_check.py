from sklearn import tree
import app.algorithm.algorithm_util as au

x_train, x_test, y_train, y_test = au.get_data(0.5)

# 用决策树分类
dt = tree.DecisionTreeClassifier(max_depth=4)
dt.fit(x_train, y_train)
predictions = dt.predict(x_test)
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(dt, out_file=f, feature_names=['recent_login', 'avg_monthly_order', 'trading_brands',
                                                            'customer_price', 'reg_time', '6_months_avg_transactions',
                                                            '6_months_avg_SKUs', '6_months_avg_transaction_amount',
                                                            'dshl_score', 'cancel_order_number'])
print(dt.feature_importances_)
for i, prediction in enumerate(predictions[-1:]):
    print('预测类型：%s. 信息: %s' % (prediction, x_test.iloc[i]))

au.validate_result(dt, x_test, y_test)

au.draw_roc(dt, x_test, y_test)

au.draw_ks(dt, x_test, y_test)
