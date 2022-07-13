
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import time

# DATASET

class TransformerDataset(Dataset):

    def __init__(self, data, tokenizer, args, no_cis=False):

        super(TransformerDataset, self).__init__()

        # Account for correcting class imbalance?
        self.data = (data if no_cis or args.cis is None
                    else data.sample(args.cis,
                                    replace=True,
                                    weights=(1 / data.groupby('Label')['Label'].transform('count'))))

        self.tokenizer = tokenizer
        self.input_max_length = args.input_max_length


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        amount, text, label = self.data.iloc[index]

        # Use tokenizer on raw text
        input = self.tokenizer.encode_plus(text,
                                           add_special_tokens=True,
                                           return_attention_mask=True,
                                           max_length=self.input_max_length,
                                           padding='max_length',
                                           truncation=True,
                                           return_tensors='pt'
                                           )

        return {
                'input_ids' : input['input_ids'].flatten(),
                'attention_mask' : input['attention_mask'].flatten(),
                'token_type_ids' : input['token_type_ids'].flatten(),
                'amount' : torch.tensor(amount),
                'label' : torch.tensor(label)
                }

    # Use to visualize token length distribution. Helped me set input_max_length = 50
    def get_token_lengths(self):

        lengths = []

        for i in range(len(self)):
            text = self.data.iloc[i, 1]
            input = self.tokenizer.encode_plus(text, add_special_tokens=True,
                                                    return_tensors='pt')

            lengths.append(len(input['input_ids'].flatten()))

        plt.hist(lengths, bins=200)
        plt.show()

# TRAINING

class FineTuner():

    def __init__(self, name, data, tokenizer, model, optimizer, criterion, args):

        # Prepare train and validation sets

        labels = sorted(list(set(data['Category'])))
        self.label_names = labels

        data['Label'] = data['Category'].map({label : i for i, label in enumerate(labels)})
        data = data.drop('Category', axis=1)


        self.train_df, self.val_df = train_test_split(data, test_size=args.val_size, random_state=args.seed)

        self.train_data = TransformerDataset(self.train_df, tokenizer, args)
        self.val_data = TransformerDataset(self.val_df, tokenizer, args, no_cis=True)

        self.train_loader = DataLoader(dataset=self.train_data, shuffle=True, batch_size=args.batch_size)
        self.val_loader = DataLoader(dataset=self.val_data, batch_size=args.batch_size)

        # Prepare training

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        if args.freeze_base_params:
            for param in self.model.base.parameters():
                param.requires_grad = False

        self.criterion = criterion.to(self.device)
        self.n_epochs = args.n_epochs
        self.n_checkpoints = args.n_checkpoints

        self.optimizer = optimizer
        self.scheduler = self.set_up_scheduler(args)

        # For saving models

        self.args = args
        self.tokenizer = tokenizer
        self.model_file = 'transformer_models/storage/' + name + '.pt'
        self.best_acc = None
        self.best_f1 = None

    # Cosine annealing with warm restarts
    def set_up_scheduler(self, args):

        if args.fixed_lr:
            return None

        return CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=int(100 * args.T_0),
            T_mult=args.T_mult,
            eta_min=args.eta_min
        )

    # Interface for human labeling :)
    def label_val_data(self, start=99, end=200):

        csv_file = '../data/human_labels/' + str(time.time()) + '.csv'
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'amount', 'text', 'label', 'prediction'])

        sample_df = self.val_df.sample(frac=1)

        errors = 0
        total = 0

        for i in range(start, end):
            amount, text, _, label = sample_df.iloc[i]
            print()
            print('Example #', i)
            print('---------------')
            print()
            print('Text:', text)
            print()
            print('Labels:', *['(' + str(i) + ') ' + label for i, label in enumerate(self.label_names)])
            print()
            pred = int(input('Guess: '))
            print()
            if pred == label:
                print('Consistent')
            else:
                print('INCONSISTENT. Label is', label)
                errors += 1
            total += 1
            print('Estimated noise level:', errors / total)
            print()
            with open(csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([i, amount, text, label, pred])

    # Main method used to fine tune transformer
    def tune(self):

        checkpoint_num = 1
        checkpoint_accs = []
        epoch_accs = []

        for epoch in range(1, self.n_epochs+1):

            print('###############')
            print('Epoch', epoch, 'of', self.n_epochs)
            print('###############')
            print()

            epoch_losses = []
            checkpoint_losses = []
            checkpoint_lrs = []

            n_iters = len(self.train_loader)
            iters_per_check = max(1, n_iters // self.n_checkpoints)

            # Main loop
            for i, batch in enumerate(tqdm(self.train_loader, total=n_iters)):

                # Compute and record loss
                loss = self.train_loop(batch, epoch, i, n_iters)
                epoch_losses.append(loss)
                checkpoint_losses.append(loss)
                checkpoint_lrs.append(self.optimizer.param_groups[0]['lr']
                                if self.scheduler is None else self.scheduler.get_last_lr()[0])

                # If appropriate to have checkpoint validation
                if i != 0 and i != n_iters - 1 and i % iters_per_check == 0:

                    check_acc, check_f1 = self.evaluate_checkpoint(num=checkpoint_num,
                                            loss=np.mean(checkpoint_losses),
                                            lrs=(checkpoint_lrs[0], checkpoint_lrs[-1]))
                    checkpoint_accs.append((check_acc, check_f1))
                    checkpoint_losses = []
                    checkpoint_lrs = []
                    checkpoint_num += 1

            # Now validate again after epoch ends
            epoch_acc, epoch_f1 = self.evaluate_epoch(num=epoch, loss=np.mean(epoch_losses))
            epoch_accs.append((epoch_acc, epoch_f1))
            checkpoint_accs.append((epoch_acc, epoch_f1))

        return {
            'max_acc' : max(checkpoint_accs, key=lambda x: x[0]),
            'epoch_accs' : epoch_accs,
            'checkpoint_accs' : checkpoint_accs
        }

    def train_loop(self, batch, epoch, i, n_iters):

        # Training mode
        self.model.train()
        self.optimizer.zero_grad()

        # Feed through network
        output = self.model(
            input_ids=batch['input_ids'].to(self.device),
            attention_mask=batch['attention_mask'].to(self.device),
            token_type_ids=batch['token_type_ids'].to(self.device),
            amount=batch['amount'].to(self.device)
        )

        labels = batch['label'].to(self.device)

        # Compute loss
        loss = self.criterion(output, labels)
        loss.backward()

        # Optimizer and LR scheduler update
        self.step(epoch, i, n_iters)

        return float(loss)

    # Adjust learning rate and parameters
    def step(self, epoch, i, n_iters):

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step(100 * ((epoch - 1) + i / n_iters))

    # Wrapper for checkpoint evaluator
    def evaluate_checkpoint(self, num, loss, lrs):

        print('***************')
        print('Checkpoint', num, 'Statistics')
        print('Avg. Train Loss:', loss)
        print('Learning Rate Interval:', list(lrs))
        acc, f1 = self.evaluate()
        print('***************')
        print()

        return acc, f1

    # Wrapper for epoch evaluator
    def evaluate_epoch(self, num, loss):

        print('---------------')
        print('Epoch', num, 'Statistics')
        print('Avg. Train Loss:', loss)
        acc, f1 = self.evaluate()
        print('---------------')
        print()

        return acc, f1

    # Main method for evaluation
    def evaluate(self, test_loader=None, return_preds=False):

        output_stats = test_loader is None
        test_loader = self.val_loader if output_stats else test_loader

        # Eval mode
        self.model.eval()

        with torch.no_grad():

            preds, labels = [], []

            # Iterate through validation set
            for batch in tqdm(test_loader, total=len(test_loader)):

                # Feed through network
                output = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device),
                    amount=batch['amount'].to(self.device)
                ).detach()

                # Add predictions and labels
                preds += output.argmax(dim=1).tolist()
                labels += batch['label'].tolist()

        preds, labels = np.array(preds), np.array(labels)

        # Compute metrics
        acc = np.mean(preds == labels)
        f1 = f1_score(labels, preds, average='macro')

        if output_stats:

            print('Val. Accuracy:', acc)
            print('Val. F1 Score:', f1)

        if return_preds:
            return list(preds)

        # Save model state if appropriate
        if test_loader == self.val_loader:
            if (self.best_acc is None or acc > self.best_acc
                        or (acc == self.best_acc and f1 > self.best_f1)):

                    self.best_acc, self.best_f1 = acc, f1
                    torch.save(self.model.state_dict(), self.model_file)
                    print('New top score. Saving model...')

        return acc, f1

    # Main method for prediction
    def predict(self, data=None, model_file=None):

        if model_file is None and self.best_acc is None:
            print('Haven\'t fine-tuned model yet')
            return

        elif model_file is None:
            model_file = self.model_file

        if data is not None:
            # Prepare data loader
            test_data = TransformerDataset(data, self.tokenizer, self.args, no_cis=True)
            test_loader = DataLoader(dataset=test_data, batch_size=self.args.batch_size)
        else:
            test_loader=None


        # Load model
        self.model.load_state_dict(torch.load(model_file))

        # Evaluate
        return self.evaluate(test_loader=test_loader, return_preds=True)

# LOSS FUNCTION

class AlphaLoss(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, labels):

        output = F.softmax(output, dim=1)
        target = F.one_hot(labels, num_classes=output.shape[1])
        P_y = torch.sum(output * target, dim=1)
        A = self.alpha / (self.alpha - 1)
        loss = torch.mean(A * (1 - P_y ** (1 / A)))
        return loss
