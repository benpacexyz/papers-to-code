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

# uv run --active --directory .\webphish python -m src.step_01_training
# __file__ not defined (e.g. PyCharm console)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
    MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models', 'web_phish_step_01')
except NameError:
    SCRIPT_DIR = os.getcwd()
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
    MODEL_DIR = os.path.join(SCRIPT_DIR, 'models', 'web_phish_step_01')

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
    'batch_size': 32,
    'embed_dim': 16,
    'conv_channels': 16,
    'kernel_size': 8,
    'learning_rate': .0015,
    'epochs': 20,
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
class WebPhishBaseModel(nn.Module):
    def __init__(self,
                 token_vocab_size,
                 token_embed_dim=16,
                 conv_out_channels=16,
                 conv_kernel_size=8,
                 max_seq_length=1898):
        super(WebPhishBaseModel, self).__init__()

        # embedding layer for raw token vectors
        self.html_embedding = nn.Embedding(
            token_vocab_size,
            token_embed_dim,
            padding_idx=0
        )

        # convolutional layer
        self.conv1_html = nn.Conv1d(
            in_channels=token_embed_dim,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size
        )

        # fully connected layers
        self.fc1 = nn.Linear(conv_out_channels * ((max_seq_length - conv_kernel_size + 1) // 2), 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        # embedding layer
        html_embed = self.html_embedding(x).permute(0, 2, 1)

        # convolution layer
        html_conv1 = F.relu(self.conv1_html(html_embed))

        # max pooling
        pooled = F.max_pool1d(html_conv1, kernel_size=2)

        # Flatten the pooled features
        flattened = pooled.view(pooled.size(0), -1)

        # Fully connected layers
        fc1_out = F.relu(self.fc1(flattened))
        fc2_out = F.relu(self.fc2(fc1_out))
        output = self.fc3(fc2_out)

        # this layer can be added to convert logits -> probability
        # output = torch.sigmoid(self.fc3(fc2_out))

        return output


##########
# ### Train Model With Selected Parameters

# initialize the model being tuned
nn_model = WebPhishBaseModel(
    token_vocab_size=len(word_to_index),
    token_embed_dim=web_phish_params['embed_dim'],
    conv_out_channels=web_phish_params['conv_channels'],
    conv_kernel_size=web_phish_params['kernel_size'],
    max_seq_length=web_phish_params['max_doc_length']
)

# send model to gpu_device
nn_model.to(gpu_device)

# initialize the adam optimizer
optimizer = torch.optim.Adam(
    nn_model.parameters(),
    lr=web_phish_params['learning_rate']
)

# create criterion / loss function
criterion = nn.BCEWithLogitsLoss()

# turn on training mode dropout=on
nn_model.train()

# train the model for epochs
for epoch in range(web_phish_params['epochs']):
    # in case loss isn't referenced
    loss = float("inf")

    # perform gradient descent for each batch of data
    for x_train, y_train in html_train_pt_load:
        # reshape y_train to be correct dim
        y_train = y_train.unsqueeze(1).float()

        # send data to gpu
        x_train, y_train = x_train.to(gpu_device), y_train.to(gpu_device)

        # make a prediction (forward-pass) using current batch
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
    print("Epoch: {0}, Loss: {1}".format(epoch, loss.item()))

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
# Model Accuracy: 0.999
# Model Precision: 0.9974
# Model Recall: 0.9995
# Model F1-Score: 0.9985
# Model False Positive Rate (FPR): 0.0012
# Model False Negative Rate (FNR): 0.0005

##########
# ### Test Model On Validation Data

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

# generate a data loader to train model
# noinspection PyTypeChecker
html_test_pt_load = DataLoader(
    dataset=html_test_tuples,
    batch_size=32
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
# Model Accuracy: 0.9524
# Model Precision: 0.9423
# Model Recall: 0.9077
# Model F1-Score: 0.9247
# Model False Positive Rate (FPR): 0.0264
# Model False Negative Rate (FNR): 0.0923

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
# Model Accuracy: 0.952
# Model Precision: 0.9554
# Model Recall: 0.8925
# Model F1-Score: 0.9229
# Model False Positive Rate (FPR): 0.0198
# Model False Negative Rate (FNR): 0.1075

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

