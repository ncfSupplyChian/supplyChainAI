from sklearn import tree
from sklearn.pipeline import Pipeline

import app.algorithm.algorithm_util as au

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = au.get_data(0.5)

    pipeline = Pipeline([
        ('clf', tree.DecisionTreeClassifier())
    ])
    parameters = {
        'clf__criterion': ('gini', 'entropy'),
        # 'clf__max_depth': (4, 5, 6, 7, 8, 9, 10),
        # 'clf__max_leaf_nodes': (None, 5, 10),
    }
    au.validate_grid(pipeline, parameters, x_train, y_train, x_test, y_test)
