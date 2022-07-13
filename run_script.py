
import os
import time
import argparse

from parameters.word_cluster_params import script_path, script_file, single_params, sbatch_params

parser = argparse.ArgumentParser(description='Run a script via sbatch or interactive mode')
parser.add_argument('--sbatch', action='store_true', help='creates sbatch for script')
parser.set_defaults(sbatch=False)
args = parser.parse_args()

if args.sbatch:

    print('Submitted', len(sbatch_params), 'jobs')

    for params in sbatch_params:
        os.system('sbatch run_script.sh \'' + script_path + ' \' \'' + script_file + '\' \'' + params + '\'')
else:
    os.chdir(script_path)
    os.system('python3 ' + script_file + ' ' + single_params)
