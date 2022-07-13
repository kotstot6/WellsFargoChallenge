
from parameters.utils import make_sbatch_params
import time
import csv

script_path = 'deep_learning/'
script_file = 'main.py'

csv_file = str(time.time()) + '.csv'

with open('deep_learning/results/' + csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['settings', 'max_acc', 'epoch_accs', 'checkpoint_accs'])


single_params = ('--val_size 0.025 --input_max_length 50 --use_amounts '
                + '--predict_model model-bert-alpha-1-input_max_length-50-use_amounts-val_size-0025-save_file-16574903780846167csv-T_0-05-lr-2e-05-seed-1.pt '
                + '--predict_dataset testing_final.csv')

sbatch_params = [   '--cis 80 --val_size 0.001 --n_checkpoints 2 --save_file ' + csv_file,
                    '--cis 90 --val_size 0.001 --n_checkpoints 2 --save_file ' + csv_file,
                    '--cis 100 --val_size 0.001 --n_checkpoints 2 --save_file ' + csv_file,
                ]

param_dict = {
    'model' : ['bert'],
    'alpha' : [1],
    'input_max_length' : [50],
    'use_amounts' : [True],
    'val_size' : [0.025],
    'save_file' : [csv_file],
    'T_0' : [0.5],
    'lr' : [2e-5],
    'seed' : [1],
    'n_epochs' : [3]
}

sbatch_params = make_sbatch_params(param_dict)
