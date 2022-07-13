
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from hyperopt import hp

params = {
    'nb' : {
        'name' : 'Multinomial Naive Bayes',
        'init' : lambda p: MultinomialNB(**p),
        'space' : {
            'fit_prior' : hp.choice('fit_prior', [True, False]),
            'alpha' : hp.uniform('alpha', 0, 1),
        },
        'point' : {
            'fit_prior' : 0, #index
            'alpha' : 0.7,
        },
        'n_iters' : 50
    },
    'knn' : {
        'name' : 'K-Nearest Neighbors',
        'init' : lambda p: KNeighborsClassifier(n_neighbors=int(p['n_neighbors']), weights=p['weights']),
        'space' : {
            'n_neighbors' : hp.quniform('n_neighbors', 1, 100, 1),
            'weights' : hp.choice('weights', ['distance', 'uniform'])
        },
        'point' : {
            'n_neighbors' : 15,
            'weights' : 0 #index
        },
        'n_iters' : 30
    },
    'rf' : {
        'name' : 'Random Forest',
        'init' : lambda p: RandomForestClassifier(n_estimators=int(p['n_estimators']),
                        min_samples_split=int(p['min_samples_split']),
                        max_features=p['max_features'], random_state=0, verbose=2),
        'space' : {
            'n_estimators' : hp.quniform('n_estimators', 1, 150, 1),
            'min_samples_split' : hp.quniform('min_samples_split', 2, 10, 1),
            'max_features' : hp.choice('max_features', ['log2', 'sqrt'])
        },
        'point' : {
            'n_estimators' : 100,
            'min_samples_split' : 2,
            'max_features' : 0 #index
        },
        'n_iters' : 50
    },
    'xgb' : {
        'name' : 'XGBoost',
        'init' : lambda p: XGBClassifier(n_estimators=50, max_depth=int(p['max_depth']),
                    gamma=p['gamma'], reg_alpha=int(p['reg_alpha']), reg_lambda=p['reg_lambda'],
                    min_child_weight=int(p['min_child_weight']), random_state=0),
        'space' : {
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'gamma': hp.uniform('gamma', 0, 2),
            'reg_alpha': hp.choice('reg_alpha', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]),
            'reg_lambda': hp.choice('reg_lambda', [0, 1e-1, 1, 5, 10, 20, 50, 100]),
            'min_child_weight' : hp.quniform('min_child_weight', 0, 5, 1),
        },
        'point' : {
            'n_estimators': 50,
            'max_depth': 6,
            'gamma': 0,
            'reg_alpha' : 0, #index
            'reg_lambda' : 2, #index
            'min_child_weight' : 1,
        },
        'n_iters' : 50
    },
    'lgbm' : {
        'name' : 'LightGBM',
        'init' : lambda p: LGBMClassifier(boosting_type=p['boosting_type'],
                        num_leaves=int(p['num_leaves']), min_child_samples=int(p['min_child_samples']),
                        subsample=p['subsample'], min_child_weight=p['min_child_weight'],
                        colsample_bytree=p['colsample_bytree'], reg_alpha=p['reg_alpha'],
                        reg_lambda=p['reg_lambda'], learning_rate=p['learning_rate'],
                        n_estimators=p['n_estimators'], random_state=0),
        'space' : {
            'boosting_type' : hp.choice('boosting_type', ['gbdt','dart']),
            'num_leaves': hp.quniform('num_leaves', 6, 50, 1),
            'min_child_samples': hp.quniform('min_child_samples', 10, 50, 1),
            'min_child_weight': hp.choice('min_child_weight', [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]),
            'subsample': hp.uniform('subsample', 0.2, 1.),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 1.),
            'reg_alpha': hp.choice('reg_alpha', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]),
            'reg_lambda': hp.choice('reg_lambda', [0, 1e-1, 1, 5, 10, 20, 50, 100]),
            'learning_rate' : hp.uniform('learning_rate', 0.05, 0.1),
            'n_estimators' : hp.choice('n_estimators', [100, 150, 200])
        },
        'point' : {
            'boosting_type' : 0, #index
            'num_leaves' : 31,
            'min_child_samples' : 20,
            'min_child_weight' : 1, #index
            'subsample': 1,
            'colsample_bytree' : 1,
            'reg_alpha' : 1, #index
            'reg_lambda' : 1, #index
            'learning_rate' : 0.1,
            'n_estimators' : 0
        },
        'n_iters' : 75
    },
    'cat' : {
        'name' : 'CatBoost',
        'init' : lambda p: CatBoostClassifier(**p, random_state=0),
        'space' : {
            'iterations' : hp.choice('iterations', [50, 100, 150, 200]),
            'learning_rate' : hp.choice('learning_rate', [1e-2, 3e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1]),
            'reg_lambda' : hp.choice('reg_lambda', [1e-2, 1e-1, 1, 2, 3, 5, 10, 20, 50, 100]),
            'bagging_temperature' : hp.uniform('bagging_temperature', 0, 10),
            'depth' : hp.quniform('depth', 3, 7, 1),
            'border_count' : hp.choice('border_count', [5, 10, 20, 32, 50, 100, 200]),
        },
        'point' : {
            'iterations' : 2, #index
            'learning_rate' : 6, #index
            'reg_lambda' : 4, #index
            'bagging_temperature' : 1,
            'depth' : 6,
            'border_count' : 3, #index
        },
        'n_iters' : 50
    },
}
