
from parameters.utils import make_sbatch_params
import time

script_path = 'machine_learning/'
script_file = 'main.py'

csv_file = str(time.time()) + '.csv'
with open('machine_learning/results/' + csv_file, 'w') as f:
    f.write('settings,best point,acc,f1\n')

single_params = '--algo xgb --bayesian_op --dataset brand-stem-tfidf-tsvd50-seed1.csv --use_amounts --use_mcode'

sbatch_params = [   '--cis 80 --val_size 0.001 --n_checkpoints 2 --save_file ' + csv_file,
                    '--cis 90 --val_size 0.001 --n_checkpoints 2 --save_file ' + csv_file,
                    '--cis 100 --val_size 0.001 --n_checkpoints 2 --save_file ' + csv_file,
                ]

param_dict = {
    'algo' : ['nb', 'knn', 'rf', 'xgb', 'lgbm', 'cat'],
    'dataset' : ['brand-stem-tfidf-tsvd50-seed1.csv', 'brand-mcat-desc-stem-tfidf-tsvd50-seed1.csv',
    'brand-mcat-desc-sep-stem-tfidf-tsvd50-seed1.csv'],
    'use_mcode' : [True],
    'use_amounts' : [True],
    'bayesian_op' : [True],
    'eval_trials' : [5],
    'save_file' : [csv_file]
}

sbatch_params = make_sbatch_params(param_dict)
