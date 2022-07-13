
from parameters.utils import make_sbatch_params

script_path = 'visuals/'
script_file = 'make_word_cluster.py'

single_params = '--max_font_size 30 --alpha 0.3 --eps1 30 --eps2 0.5 --max_words 500 --svg'

sbatch_params = [   '--cis 80 --val_size 0.001 --n_checkpoints 2 --save_file ',
                    '--cis 90 --val_size 0.001 --n_checkpoints 2 --save_file ',
                    '--cis 100 --val_size 0.001 --n_checkpoints 2 --save_file ',
                ]

param_dict = {
    'max_font_size' : [30, 40, 50],
    'alpha' : [0.25, 0.3, 0.35],
    'eps1' : [30, 50, 100],
    'eps2' : [0.25, 0.5, 1.],
    'max_words' : [500]
}

sbatch_params = make_sbatch_params(param_dict)
