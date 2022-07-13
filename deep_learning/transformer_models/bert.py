
"""
Implementation of BERT transformer
by Kyle Otstot
"""

import transformers
import torch
import torch.nn as nn
import torch.optim as optim

MODEL_NAME = 'bert-base-uncased'

# TOKENIZER

tokenizer = transformers.BertTokenizer.from_pretrained(MODEL_NAME)

# MODEL

class Model(nn.Module):

    def __init__(self, use_amounts=False, hidden_layer=False):

        super(Model, self).__init__()

        self.base = transformers.BertModel.from_pretrained(MODEL_NAME)

        if hidden_layer:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(769, 100) if use_amounts else nn.Linear(768, 100),
                nn.Dropout(p=0.1),
                nn.Linear(100, 10)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(769, 10) if use_amounts else nn.Linear(768, 10)
            )

        self.use_amounts = use_amounts

    def forward(self, input_ids, attention_mask, token_type_ids, amount):

        _ , base_output = self.base(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        return_dict=False)

        if self.use_amounts:
            amount = amount.reshape(-1,1).type(base_output.dtype)
            classifier_input = torch.cat((base_output, amount), dim=1)
        else:
            classifier_input = base_output

        logits = self.classifier(classifier_input)

        return logits

# OPTIMIZER

def optimizer(model, lr):
    return optim.AdamW(model.parameters(), lr=lr)

# UTILITY

def get_items(args):

    model = Model(use_amounts=args.use_amounts, hidden_layer=args.hidden_layer)

    return {
        'tokenizer' : tokenizer,
        'model' : model,
        'optimizer' : optimizer(model, lr=args.lr)
    }
