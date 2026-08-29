##########
# ### Import Packages and Setup Paths

# standard
import os
import re
import json
import pandas as pd

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# utils
import src.utils as ut

# uv run --active --directory .\webphish python -m src.step_02_training_auto
# __file__ not defined (e.g. PyCharm console)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
    MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models', 'web_phish_step_02')
except NameError:
    SCRIPT_DIR = os.getcwd()
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
    MODEL_DIR = os.path.join(SCRIPT_DIR, 'models', 'web_phish_step_02')

##########
# ### Initial Setup

# set the device to the available gpu, else fall back to cpu
ut.print_cuda_names()
if torch.cuda.is_available():
    gpu_device = torch.device("cuda:0")
else:
    gpu_device = torch.device("cpu")

# minimum number of times a word must appear in corpus to
# be included in word index (derived below)
MIN_WORD_COUNT = 10

##########
# ### Dictionary of Hyperparameters

# training parameters
web_phish_params = {
    'batch_size': 128,
    'embed_dim': 32,
    'conv_channels': 64,
    'kernel_sizes': [3, 5, 8],
    'learning_rate': 0.0027,  # derived from optuna trial
    'weight_decay': .0001,
    'dropout': .3,
    'pos_weight': 2.0,
    'epochs': 7,
    'max_doc_length': 1898
}

##########
# ### Load in and Prepare Data for Training

# load in html training data
html_train_df = pd.read_parquet(
    os.path.join(DATA_DIR, 'train-test', 'html_train.parquet')
)

# truncate html before tokenization
html_train_df['html_word_parsed'] = html_train_df['html_word_parsed'].apply(
    func=lambda x: x[0:web_phish_params['max_doc_length']]
)

# identify pure duplicates and drop them
html_train_df = ut.de_dupe_html_by_words(
    html_df=html_train_df
)

# generate word count to drop rare words
train_word_counts_df = ut.get_word_count_df(
    html_parsed_docs=list(html_train_df['html_word_parsed'])
)

# subset to words with greater than 10 word count
train_valid_tokens = train_word_counts_df[
    (train_word_counts_df['count'] >= MIN_WORD_COUNT)
]['word'].tolist()

# get word index
word_to_index = ut.get_word_to_index(
    html_parsed_list=train_valid_tokens
)

# get parsed html as tokens
html_train_df['html_word_tokens'] = html_train_df['html_word_parsed'].apply(
    func=ut.get_html_tokens,
    word_to_index=word_to_index,
    max_doc_len=web_phish_params['max_doc_length']
)

# convert df into pytorch tuples
html_train_tuples = [
    (torch.tensor(row.html_word_tokens, dtype=torch.long), row.phishing)
    for row in html_train_df.itertuples()
]

# generate a data loader to train model
# noinspection PyTypeChecker
html_train_pt_load = DataLoader(
    dataset=html_train_tuples,
    batch_size=web_phish_params['batch_size'],
    shuffle=True
)


##########
# ### Define WebPhish Model Architecture

# web phish model architecture
class WebPhishModel(nn.Module):
    def __init__(self,
                 token_vocab_size,
                 token_embed_dim=32,
                 conv_out_channels=64,
                 conv_kernel_sizes=(3, 5, 8),
                 dropout=.3):
        super(WebPhishModel, self).__init__()

        # embedding layer for raw token vectors
        self.html_embedding = nn.Embedding(
            token_vocab_size,
            token_embed_dim,
            padding_idx=0
        )

        # parallel convolutional layers
        self.html_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=token_embed_dim,
                out_channels=conv_out_channels,
                kernel_size=kernel_size
            )
            for kernel_size in conv_kernel_sizes
        ])

        # fully connected layers
        self.fc1 = nn.Linear(
            conv_out_channels * len(conv_kernel_sizes),
            32
        )
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # embedding layer
        html_embed = self.html_embedding(x).permute(0, 2, 1)

        # convolutional layers with global max pooling
        html_pooled = [
            torch.amax(F.relu(conv(html_embed)), dim=2)
            for conv in self.html_convs
        ]

        # concatenate pooled convolutional features
        html_features = torch.cat(html_pooled, dim=1)

        # fully connected layers
        fc1_out = self.dropout(F.relu(self.fc1(html_features)))
        fc2_out = self.dropout(F.relu(self.fc2(fc1_out)))
        output = self.fc3(fc2_out)

        return output


##########
# ### Train Model With Selected Parameters

# initialize the model being trained
nn_model = WebPhishModel(
    token_vocab_size=len(word_to_index),
    token_embed_dim=web_phish_params['embed_dim'],
    conv_out_channels=web_phish_params['conv_channels'],
    conv_kernel_sizes=web_phish_params['kernel_sizes'],
    dropout=web_phish_params['dropout']
)

# send model to gpu_device
nn_model.to(gpu_device)

# initialize the adamw optimizer
optimizer = torch.optim.AdamW(
    nn_model.parameters(),
    lr=web_phish_params['learning_rate'],
    weight_decay=web_phish_params['weight_decay']
)

# create criterion / loss function
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [web_phish_params['pos_weight']],
        device=gpu_device
    )
)

# turn on training mode dropout=on
nn_model.train()

# train the model for seven epochs
for epoch in range(web_phish_params['epochs']):
    # in case loss isn't referenced
    loss = float('inf')

    # perform gradient descent for each batch of data
    for x_train, y_train in html_train_pt_load:
        # reshape y_train to be correct dim
        y_train = y_train.unsqueeze(1).float()

        # send data to gpu
        x_train, y_train = x_train.to(gpu_device), y_train.to(gpu_device)

        # make a prediction using current batch
        y_hat = nn_model(x_train)

        # calculate error/loss
        loss = criterion(y_hat, y_train)

        # zero out the gradient before differentiation
        optimizer.zero_grad()

        # calculate weight adjustments
        loss.backward()

        # update weights based on gradient
        optimizer.step()

    # print out epoch and loss
    print('Epoch: {0}, Loss: {1}'.format(epoch, loss.item()))

# evaluate on training set
print('Training WebPhish Results:')
html_train_pred = ut.evaluate_nn(
    trained_model=nn_model,
    test_loader=html_train_pt_load,
    gpu_device=gpu_device
)
ut.print_score_metrics(
    y_true=html_train_pred[0],
    y_pred=html_train_pred[1]
)
print()
# Training WebPhish Results:
# Model Accuracy: 0.9953
# Model Precision: 0.9905
# Model Recall: 0.9945
# Model F1-Score: 0.9925
# Model False Positive Rate (FPR): 0.0044
# Model False Negative Rate (FNR): 0.0055

##########
# ### Test Model On Testing Data

# load in html testing data
html_test_df = pd.read_parquet(
    os.path.join(DATA_DIR, 'train-test', 'html_test.parquet')
)

# get parsed html as tokens
html_test_df['html_word_tokens'] = html_test_df['html_word_parsed'].apply(
    func=ut.get_html_tokens,
    word_to_index=word_to_index,
    max_doc_len=web_phish_params['max_doc_length']
)

# convert df into pytorch tuples
html_test_tuples = [
    (torch.tensor(row.html_word_tokens, dtype=torch.long), row.phishing)
    for row in html_test_df.itertuples()
]

# generate a data loader to test model
# noinspection PyTypeChecker
html_test_pt_load = DataLoader(
    dataset=html_test_tuples,
    batch_size=web_phish_params['batch_size']
)

# evaluate on testing set
print('Testing WebPhish Results:')
html_test_pred = ut.evaluate_nn(
    trained_model=nn_model,
    test_loader=html_test_pt_load,
    gpu_device=gpu_device
)
ut.print_score_metrics(
    y_true=html_test_pred[0],
    y_pred=html_test_pred[1]
)
print()
# Testing WebPhish Results:
# Model Accuracy: 0.9672
# Model Precision: 0.9485
# Model Recall: 0.9498
# Model F1-Score: 0.9492
# Model False Positive Rate (FPR): 0.0245
# Model False Negative Rate (FNR): 0.0502

##########
# ### Validate Model On Final Holdout Data

# load in html validation data
html_valid_df = pd.read_parquet(
    os.path.join(DATA_DIR, 'train-test', 'html_valid.parquet')
)

# get parsed html as tokens
html_valid_df['html_word_tokens'] = html_valid_df['html_word_parsed'].apply(
    func=ut.get_html_tokens,
    word_to_index=word_to_index,
    max_doc_len=web_phish_params['max_doc_length']
)

# convert df into pytorch tuples
html_valid_tuples = [
    (torch.tensor(row.html_word_tokens, dtype=torch.long), row.phishing)
    for row in html_valid_df.itertuples()
]

# generate a data loader to validate model
# noinspection PyTypeChecker
html_valid_pt_load = DataLoader(
    dataset=html_valid_tuples,
    batch_size=web_phish_params['batch_size']
)

# evaluate on final validation set
print('Validation WebPhish Results:')
html_valid_pred = ut.evaluate_nn(
    trained_model=nn_model,
    test_loader=html_valid_pt_load,
    gpu_device=gpu_device
)
ut.print_score_metrics(
    y_true=html_valid_pred[0],
    y_pred=html_valid_pred[1]
)
print()
# Validation WebPhish Results:
# Model Accuracy: 0.9627
# Model Precision: 0.9425
# Model Recall: 0.9416
# Model F1-Score: 0.942
# Model False Positive Rate (FPR): 0.0273
# Model False Negative Rate (FNR): 0.0584

##########
# ### Save Model Weights and Word Index

# add in today's date/time
current_date_time = pd.Timestamp.now()
current_date_time_str = str(current_date_time)[0:19]
current_date_time_str = re.sub(r':', '', current_date_time_str)
current_date_time_str = re.sub(r' ', '-', current_date_time_str)

# ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# save pytorch model weights
torch.save(
    nn_model.state_dict(),
    os.path.join(MODEL_DIR, 'web-phish-weights-' + current_date_time_str + '.pth')
)

# save word to index dictionary
with open(os.path.join(MODEL_DIR, 'web-phish-word-index-' + current_date_time_str + '.json'), 'w') as json_file:
    json.dump(word_to_index, json_file)
