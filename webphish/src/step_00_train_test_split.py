##########
# ### Import Packages and Setup Paths

# standard
import os
import glob
import numpy as np
import pandas as pd

# machine learning
from sklearn.model_selection import train_test_split

# utils
import src.utils as ut

# __file__ not defined (e.g. PyCharm console)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# setup data directory relative to this file
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

##########
# ### Load In Collected Data and WebPhish Data

# load in collected data
collected_html_df = pd.concat(
    objs=[
        pd.read_parquet(p)
        for p in sorted(glob.glob(os.path.join(DATA_DIR, 'raw', 'collected_*.parquet')))
    ],
    ignore_index=True
)

# drop index column carried over from the original pickle
collected_html_df = collected_html_df.drop(columns=['orig_index'])

# load in web phish data
web_phish_html_df = pd.concat(
    objs=[
        pd.read_parquet(p)
        for p in sorted(glob.glob(os.path.join(DATA_DIR, 'raw', 'webphish_*.parquet')))
    ],
    ignore_index=True
)
web_phish_html_df.columns = ['phishing', 'html_content']

# convert to binary label 0 or 1
web_phish_html_df['phishing'] = np.where(
    (web_phish_html_df['phishing'] == 'spam'),
    1,
    0
)

# get parsed html content
web_phish_html_df['html_char_parsed'] = web_phish_html_df['html_content'].apply(
    func=ut.get_html_char_parsed
)
web_phish_html_df['html_word_parsed'] = web_phish_html_df['html_content'].apply(
    func=ut.get_html_word_parsed
)

# fill in missing fields from collected data
web_phish_html_df['file_source'] = 'opara_et_al_web_phish_paper'
web_phish_html_df['download_type'] = 'html_available_in_data'
web_phish_html_df['date_downloaded'] = pd.to_datetime('2024-02-01', format='%Y-%m-%d')

# add in missing fields from collected data
for col in ['url', 'file_name']:
    web_phish_html_df[col] = np.nan

# reorder columns
web_phish_html_df = web_phish_html_df[
    ['url', 'file_name', 'file_source', 'download_type',
     'date_downloaded', 'html_char_parsed', 'html_word_parsed',
     'phishing']
].copy()

# merge all available data
html_df = pd.concat(
    objs=[collected_html_df, web_phish_html_df],
    ignore_index=True
)

##########
# ### Train Test Split, Save Training and Test Data

# hold out a final validation set, untouched by training or tuning
html_df, html_valid_df = train_test_split(
    html_df,
    test_size=.10,
    stratify=html_df['phishing']
)

# train test split
html_train_df, html_test_df = train_test_split(
    html_df,
    test_size=len(html_df.index)**.75 / len(html_df.index),  # calculate the train test split proportion
    stratify=html_df['phishing']
)

# ensure train-test directory exists
os.makedirs(os.path.join(DATA_DIR, 'train-test'), exist_ok=True)

# write training and test data as .parquet files
html_train_df.to_parquet(
    path=os.path.join(DATA_DIR, 'train-test', 'html_train.parquet'),
    index=False
)
html_test_df.to_parquet(
    path=os.path.join(DATA_DIR, 'train-test', 'html_test.parquet'),
    index=False
)
html_valid_df.to_parquet(
    path=os.path.join(DATA_DIR, 'train-test', 'html_valid.parquet'),
    index=False
)
