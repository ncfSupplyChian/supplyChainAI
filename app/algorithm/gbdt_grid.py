from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

import app.algorithm.algorithm_util as au

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = au.get_data(0.5)

    pipeline = Pipeline([
        ('clf', GradientBoostingClassifier(learning_rate=0.05, n_estimators=120, max_depth=3,
                                           min_samples_leaf=90, min_samples_split=700, subsample=0.85
                                           ))
    ])
    parameters = {
        # 'clf__loss': ('deviance', 'exponential'),
        # 'clf__n_estimators': range(20, 81, 10),
        # 'clf__max_depth': range(3, 14, 2),
        # 'clf__min_samples_split': range(100, 801, 200),
        # 'clf__min_samples_leaf': range(60, 101, 10)
        # 'clf__max_leaf_nodes': (None, 5, 10),
        # 'clf__subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
    }
    au.validate_grid(pipeline, parameters, x_train, y_train, x_test, y_test)
