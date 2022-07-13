
def title():

    print('++++++++++++++++++++')
    print('Script: main.py')
    print('Description: implementation of deep transformers to categorize transactions')
    print('Author: Kyle Otstot')
    print('++++++++++++++++++++')
    print()

import pandas as pd
import transformer_models.bert as bert
import transformer_models.xlnet as xlnet
from train import FineTuner, AlphaLoss
import torch.nn as nn
import sys
import argparse
import torch
import numpy as np
import random
import csv

def get_args():

    settings = ' '.join(sys.argv[1:])

    print()
    print('===== Settings =====')
    print(settings)
    print('====================')
    print()

    parser = argparse.ArgumentParser(description='deep transformers for transaction categorization')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--save_file', type=str, default=None, help='file to save results')

    # Dataset settings
    parser.add_argument('--val_size', type=float, default=0.05, help='proportion of dataset used for validation')
    parser.add_argument('--cis', type=int, default=None, help='corrected train set size')
    parser.add_argument('--input_max_length', type=int, default=100, help='max length of input sequences')
    parser.add_argument('--use_amounts', action='store_true', help='includes \'amount\' attribute in training')
    parser.set_defaults(use_amounts=False)

    # Network & Train settings
    parser.add_argument('--model', type=str, default='bert', help='model to be fine-tuned')
    parser.add_argument('--freeze_base_params', action='store_true', help='freeze the parameters in the base transformer')
    parser.add_argument('--hidden_layer', action='store_true', help='include hidden layer in classification block')
    parser.add_argument('--alpha', type=float, default=1, help='\'alpha\' parameter in AlphaLoss function')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--n_epochs', type=int, default=2, help='number of epochs for training')
    parser.add_argument('--n_checkpoints', type=int, default=10, help='number of validation checkpoints per epoch')
    parser.set_defaults(freeze_base_params=False, hidden_layer=False)

    # Cosine Annealing w/ Warm Restarts
    parser.add_argument('--fixed_lr', action='store_true', help='uses constant learning rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--eta_min', type=float, default=0, help='final learning rate')
    parser.add_argument('--T_0', type=float, default=1, help='number of EPOCHS after first restart')
    parser.add_argument('--T_mult', type=int, default=1, help='factor increases T_i after restart')
    parser.set_defaults(fixed_lr=False)

    # Predict instead of train?
    parser.add_argument('--predict_model', type=str, default=None, help='file of stored model to make predictions')
    parser.add_argument('--predict_dataset', type=str, default=None, help='dataset to make predictions on')

    return parser.parse_args(), settings

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_name(settings):
    return settings.replace(' ', '-').replace('--', '-').replace('--', '-')

def main(args, settings):

    # Training data
    data = pd.read_csv('../data/training_final.csv')

    model = xlnet if args.model == 'xlnet' else bert

    items = model.get_items(args=args)

    # Pass everything to fine tuner class
    tuner = FineTuner(
        name=name,
        data=data,
        tokenizer=items['tokenizer'],
        model=items['model'],
        optimizer=items['optimizer'],
        criterion=nn.CrossEntropyLoss() if args.alpha == 1 else AlphaLoss(args.alpha),
        args=args
        )

    # if predict model is given, then predict instead of train
    if args.predict_model is not None:

        if args.predict_dataset is not None:
            test_data = pd.read_csv('../data/' + args.predict_dataset)
            test_data['Label'] = [0] * len(test_data)
            test_data = test_data.drop('Category', axis=1)
        else:
            test_data = None

        preds = tuner.predict(data=test_data, model_file='transformer_models/storage/-' + args.predict_model)

        df_preds = pd.DataFrame({'Category' : [tuner.label_names[p] for p in preds]})
        df_preds.to_csv('predictions/' + args.predict_model.replace('.', '-') + '.csv', index=False)

        return

    # if not, then fine tune
    results = tuner.tune()

    # write results
    if args.save_file is not None:
        with open('results/' + args.save_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([name, str(results['max_acc']), str(results['epoch_accs']), str(results['checkpoint_accs'])])

if __name__ == '__main__':

    args, settings = get_args()
    name = get_name(settings)
    set_seed(args.seed)
    main(args, name)
