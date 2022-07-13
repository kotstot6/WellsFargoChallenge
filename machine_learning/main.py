
def title():

    print('++++++++++++++++++++')
    print('Script: main.py')
    print('Description: implementation of machine learning algorithms to categorize transactions')
    print('Author: Kyle Otstot')
    print('++++++++++++++++++++')
    print()

import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from hyperopt import hp, space_eval, fmin, tpe, STATUS_OK
from algo_params import params

def get_args():

    settings = ' '.join(sys.argv[1:])

    print()
    print('===== Settings =====')
    print(settings)
    print('====================')
    print()

    parser = argparse.ArgumentParser(description='machine learning algorithms for transaction categorization')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--save_file', type=str, default=None, help='file to save results')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default=None, help='dataset used for training')
    parser.add_argument('--val_size', type=float, default=0.025, help='proportion of dataset used for validation')
    parser.add_argument('--cis', type=int, default=None, help='corrected train set size')
    parser.add_argument('--use_amounts', action='store_true', help='includes \'amount\' attribute in training')
    parser.add_argument('--use_mcode', action='store_true', help='includes \'merchant code\' attribute in training')

    # Algorithms, optimization, & evaluation
    parser.add_argument('--algo', type=str, default='nb', choices={'nb', 'knn', 'rf', 'xgb', 'lgbm', 'cat'},
                                                help='ML algorithm used for transaction categorization')
    parser.add_argument('--bayesian_op', action='store_true', help='run bayesian optimization before evaluation')
    parser.add_argument('--eval_trials', type=int, default=1, help='number of trials (seeds) to test algo on')

    parser.set_defaults(use_amounts=False, use_mcode=False, bayesian_op=False)

    return parser.parse_args(), settings

def set_seed(seed):
    np.random.seed(seed)

# Prepare the train and validation sets

def make_dataset(args):

    PATH_TO_DATA = '../visuals/data/datasets/'

    if args.dataset is None:
        print('Error: must include dataset path')
        exit()

    df_raw = pd.read_csv('../data/training_reduced.csv')
    log_amount = np.log(df_raw['Amount ($)'])
    log_norm_amount = (log_amount - log_amount.mean()) / log_amount.std()
    mcode_null = df_raw['Merchant Code'].apply(lambda x: 1 if pd.isnull(x) else 0)
    mcode = df_raw['Merchant Code'].apply(lambda x: x if pd.notnull(x) else 0)

    sparse = args.dataset[-6:] == 'pickle'

    if sparse:

        with open(PATH_TO_DATA + args.dataset, 'rb') as f:
            data = pickle.load(f)

        bag, y = csr_matrix(data['bag']), data['y']

        matrices = []

        if args.use_amounts and args.algo != 'nb':
            matrices.append(csr_matrix(log_norm_amount).reshape(-1,1))

        if args.use_mcode and args.algo != 'nb':
            matrices += [csr_matrix(mcode_null).reshape(-1,1), csr_matrix(mcode).reshape(-1,1)]

        matrices.append(bag)

        X = hstack(tuple(matrices))

        return X, y, True

    data = pd.read_csv(PATH_TO_DATA + args.dataset)

    if args.use_amounts:
        data['amount'] = log_norm_amount

    if args.use_mcode:
        data['mcode_null'] = mcode_null
        data['mcode'] = mcode

    return data.drop('y', axis=1), data['y'], False

# Split up datasets into train and validation

def prepare_sets(X, y, args, sparse, seed=None):

    seed = args.seed if seed is None else seed

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.val_size, random_state=seed)

    if args.cis is not None:

        y_train = y_train.reset_index(drop=True)
        y_train = y_train.sample(args.cis, replace=True, weights=1 / y_train.groupby(y_train).transform('count'))
        X_train = X_train[y_train.index, :] if sparse else X_train.iloc[y_train.index, :]

    return X_train, X_test, y_train, y_test

# Objective function for bayesian optimization

def objective(point, algo_init, X_train, y_train, X_test, y_test, eval=False):

    print('Point:', point)

    clf = algo_init(point)

    try:
        clf.fit(X_train, y_train, eval_set=(X_test, y_test))
    except:
        clf.fit(X_train, y_train)

    preds = clf.predict(X_test).reshape(-1)

    acc = np.mean(y_test == preds)
    f1 = f1_score(y_test, preds, average='macro')

    print ('Val. Accuracy:', acc)
    print('F1 Score:', f1)
    print('---------------')
    print()

    if eval:
        return acc, f1

    return {'loss': -acc, 'status': STATUS_OK }

def main(args, settings):

    X, y, sparse = make_dataset(args)

    X_train, X_test, y_train, y_test = prepare_sets(X, y, args, sparse)

    algo_dict = params[args.algo]

    print('ML Algorithm:', algo_dict['name'])

    if args.bayesian_op:

        print('Bayesian Optimization...', algo_dict['name'])

        # FMin function uses bayesian optimization to find argminimizer of error

        best_point = fmin(fn=lambda x: objective(x, algo_dict['init'],
                            X_train, y_train, X_test, y_test),
                            space=algo_dict['space'],
                            points_to_evaluate=[algo_dict['point']],
                            max_evals=algo_dict['n_iters'],
                            algo=tpe.suggest,
                            rstate=np.random.default_rng(args.seed)
                        )

    else:
        best_point = algo_dict['point']

    best_point = space_eval(algo_dict['space'], best_point)

    print('Point to be evaluated:', best_point)

    # Evaluation time

    if args.eval_trials > 0:

        accs, f1s = [], []

        for i in range(1, args.eval_trials+1):

            print('---------------')
            print('Trial', i)
            X_train, X_test, y_train, y_test = prepare_sets(X, y, args, sparse, seed=i)

            acc, f1 = objective(best_point, algo_dict['init'],
                                        X_train, y_train, X_test, y_test, eval=True)
            accs.append(acc)
            f1s.append(f1)

        # Average over metrics

        acc_mean, acc_std = np.mean(accs), np.std(accs)
        f1_mean, f1_std = np.mean(f1s), np.std(f1s)

        acc_str = str(acc_mean) + ' ± ' + str(acc_std)
        f1_str = str(f1_mean) + ' ± ' + str(f1_std)

        print('***************')
        print('Results:')
        print('Accuracy:', acc_str)
        print('F1 Score:', f1_str)

        if args.save_file is not None:

            with open('results/' + args.save_file, 'a') as f:
                f.write('\"' + settings + '\",\"' + str(best_point) + '\",' + acc_str + ',' + f1_str + '\n')

if __name__ == '__main__':

    title()
    args, settings = get_args()
    set_seed(args.seed)
    main(args, settings)
