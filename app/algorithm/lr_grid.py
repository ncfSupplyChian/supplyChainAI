from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline

import app.algorithm.algorithm_util as au

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = au.get_data(0.5)
    pipeline = Pipeline([
        ('clf', LogisticRegression())
    ])
    parameters = {
        'clf__penalty': ('l1', 'l2'),
        # 'clf__C': (0.01, 0.1, 1, 10),
    }
    au.validate_grid(pipeline, parameters, x_train, y_train, x_test, y_test)
