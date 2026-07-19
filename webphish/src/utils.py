##########
# ### Import Packages and Setup Paths

# standard
import re
import json
import numpy as np
import pandas as pd
from collections import Counter

# data visualization
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# machine learning
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

##########
# ### PyTorch Utils

# function that prints cuda names and index
def print_cuda_names():
    # check if cuda is available
    if torch.cuda.is_available():
        # print cuda is available
        print('Cuda is available.')

        # iterate through all devices
        for device_index in range(torch.cuda.device_count()):
            print('Cuda: {0}, Name: {1}'.format(
                device_index,
                torch.cuda.get_device_name(device_index)
            ))

    # else print cuda is not available
    else:
        print('Cuda not available.')


##########
# ### Data Formatting Utils

# function to return a tokenized version of raw html
def get_html_word_parsed(html_content: str) -> list:
    # pattern to split html content by punctuation
    html_word_pattern = re.compile(
        r'[A-Za-z0-9]+|[^\w\s]'
    )

    # split raw html content using pattern
    html_parsed = html_word_pattern.findall(html_content)
    html_parsed = [x.lower() for x in html_parsed]

    # return the parsed html as a list
    return html_parsed


# function to load word_index from .json file
def load_word_index(file_path: str) -> dict:
    # load word index from file
    with open(file_path, 'r') as json_file:
        word_index_temp = json.load(json_file)

    # return word index
    return word_index_temp


# function to return a tokenized version of raw html
def get_html_char_parsed(html_content: str) -> list:
    # split raw html content by characters
    html_parsed = list(html_content)

    # return the parsed html as a list
    return html_parsed


# function to return a word to index dictionary
def get_word_to_index(html_parsed_list: list) -> dict:
    # get unique tokens by adding to a set
    unique_tokens = set(tok for tok in html_parsed_list)

    # crate a word to index dict
    word_to_index_temp = {word: i + 2 for i, word in enumerate(unique_tokens)}

    # add special padding token
    word_to_index_temp['<PAD>'] = 0
    word_to_index_temp['<UNK>'] = 1

    # return the word to index dict
    return word_to_index_temp


# function to convert parsed html to tokens
def get_html_tokens(html_parsed: list, word_to_index: dict, max_doc_len: int) -> list:
    # initialize a list to store html content
    html_fmt = [x for x in html_parsed]

    # if greater than max length truncate
    if len(html_parsed) > max_doc_len:
        html_fmt = html_fmt[0:max_doc_len]

    # else add padding token
    else:
        html_fmt.extend(['<PAD>' for x in range(max_doc_len - len(html_parsed))])

    # initialize to store html tokens
    html_tokens = []

    # convert parsed html into tokens
    for p in html_fmt:
        # token is present in index add index
        if p in word_to_index.keys():
            html_tokens.append(word_to_index[p])
        # else use reserve <UNK> token
        else:
            html_tokens.append(word_to_index['<UNK>'])

    # return html tokens
    return html_tokens


# function to get words and presence count in training data
def get_word_count_df(html_parsed_docs: list) -> pd.DataFrame:
    # initialize dict to return
    # word_count_dict = dict()

    # # iterate through all docs in html_parsed_docs list
    # for doc in html_parsed_docs:
    #     for tok in doc:
    #         if tok in word_count_dict.keys():
    #             word_count_dict[tok] += 1
    #         else:
    #             word_count_dict[tok] = 1

    # ugly but fast list comprehension, see code above for logic
    word_count_dict = Counter(
        tok for doc in html_parsed_docs for tok in doc
    )

    # convert to dataframe and return
    word_count_df = pd.DataFrame(
        list(word_count_dict.items()),
        columns=['word', 'count']
    )

    # return loaded dataframe
    return word_count_df


# drops pure duplicate from an html dataframe
def de_dupe_html_by_words(html_df: pd.DataFrame) -> pd.DataFrame:
    # throw a copy of the dataframe on the stack
    html_temp_df = html_df.copy()

    # join word parsed html to identify duplicates
    html_temp_df['html_word_joined'] = html_temp_df['html_word_parsed'].apply(
        func=lambda x: '-'.join(x)
    )

    # drop any uncaught duplicated urls
    html_temp_df = html_temp_df.drop_duplicates(
        subset='html_word_joined', keep='first'
    )

    # drop joined column before returning
    html_temp_df = html_temp_df.drop(columns=['html_word_joined'])

    # return de-duped dataframe
    return html_temp_df


# function to display distribution of tokens
def display_distribution(int_list: list, title: str):
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(int_list, bins=50, alpha=0.75, edgecolor='black')

    # Add title and labels
    plt.title(title + 'Distribution')
    plt.xlabel(title)
    plt.ylabel('Frequency')

    # Show plot
    plt.grid(True)
    plt.show()


##########
# ### WebPhish Model Training Utils

# function to get predictions
def evaluate_nn(trained_model: nn.Module, test_loader: DataLoader, gpu_device, score_thresh: float = .5) -> tuple:
    # ensure the model is in eval mode
    trained_model.eval()

    # initialize list of true and hat
    y_true_pt = list()
    y_hat_pt = list()

    # disable gradient calculation
    with torch.no_grad():

        # iterate through all test batches
        for x_test, y_test in test_loader:
            # reshape y_train to be correct dim
            y_test = y_test.unsqueeze(1).float()

            # send data to gpu
            x_test, y_test = x_test.to(gpu_device), y_test.to(gpu_device)

            # get probabilities and labels
            pred_logit_pt = trained_model(x_test)
            pred_pt = torch.sigmoid(pred_logit_pt)

            # get the predicted values
            pred_labels = (pred_pt >= score_thresh).int()
            pred_probs = pred_pt

            # push data back to cpu
            y_true_i = y_test.to("cpu")
            y_hat_i = pred_labels.to("cpu")

            # append to eval lists
            y_true_pt.extend(y_true_i)
            y_hat_pt.extend(y_hat_i)

    # convert pt tensor to python list
    y_true = [int(i.item()) for i in y_true_pt]
    y_hat = [int(i.item()) for i in y_hat_pt]

    # return the tuple list
    return y_true, y_hat


# function that prints out accuracy and f1-score
def print_score_metrics(y_true: list, y_pred: list) -> None:
    # print out accuracy, precision, recall, and f1-score
    print('Model Accuracy: {0}'.format(
        round(accuracy_score(
            y_true=y_true,
            y_pred=y_pred
        ), 4)
    ))
    print('Model Precision: {0}'.format(
        round(precision_score(
            y_true=y_true,
            y_pred=y_pred
        ), 4)
    ))
    print('Model Recall: {0}'.format(
        round(recall_score(
            y_true=y_true,
            y_pred=y_pred
        ), 4)
    ))
    print('Model F1-Score: {0}'.format(
        round(f1_score(
            y_true=y_true,
            y_pred=y_pred
        ), 4)
    ))

    # calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # calculate false positive rate (fpr) and false negative rate (fnr)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    # print the false positive and false negative rate
    print('Model False Positive Rate (FPR): {0}'.format(
        round(fpr, 4)
    ))
    print('Model False Negative Rate (FNR): {0}'.format(
        round(fnr, 4)
    ))


##########
# ### BERT Fine Tuning Functions

# function to tokenize hf dataset
def bert_tokenize(rows, tokenizer):
    # tokenize rows and return
    rows_tokenized = tokenizer(
        rows['text'],
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # return tokenized data
    return rows_tokenized


# function that returns metrics for bert model
def get_bert_score_metrics(bert_pred):
    # extract predictions and true labels
    y_pred, y_true = bert_pred

    # get the predicted class by finding the index of the maximum logit
    y_pred = np.argmax(y_pred, axis=1)

    # calculate metrics
    accuracy = round(accuracy_score(y_true=y_true, y_pred=y_pred), 4)
    precision = round(precision_score(y_true=y_true, y_pred=y_pred), 4)
    recall = round(recall_score(y_true=y_true, y_pred=y_pred), 4)
    f1 = round(f1_score(y_true=y_true, y_pred=y_pred), 4)

    # calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # calculate false positive rate (fpr) and false negative rate (fnr)
    fpr = round(fp / (fp + tn), 4)
    fnr = round(fn / (fn + tp), 4)

    # return a dictionary of the calculated metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr
    }

    return metrics