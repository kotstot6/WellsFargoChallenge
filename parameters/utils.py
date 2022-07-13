
"""
Helps turn dictionary into a list of sbatch arguments
by Kyle Otstot
"""

from itertools import product

def make_sbatch_params(param_dict):

    trials = [ { p : t for p, t in zip(param_dict.keys(), trial) }
                    for trial in list(product(*param_dict.values())) ]

    def trial_to_args(trial):
        arg_list = ['--' + param + ' ' + str(val) if type(val) != type(True)
                else '--' + param if val else '' for param, val in trial.items()]
        return ' '.join(arg_list)

    sbatch_params = [trial_to_args(trial) for trial in trials]

    return sbatch_params
