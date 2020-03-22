# loading packages
import pandas as pd
import numpy as np

# setting seed for reproducability
np.random.seed(42)

# loading dataframes and adapting labels and headers

# setting the urls where to get the data from github
url_davidson = "https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/data/twitter%20data/davidson%20et%20al./labeled_data.csv"
url_founta = "https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/data/twitter%20data/founta%20et%20al./hatespeech_text_label_vote.csv"


data_davidson = pd.read_csv(url_davidson, names=["count", "hate_speech", "offensive_language", "neither", "label", "tweet"], header=1)
data_founta = pd.read_csv(url_founta, sep='\t', names=["tweet", "label_text", "count"])

data_founta = data_founta[~data_founta.label_text.str.contains("spam")]
data_founta['label'] = data_founta.label_text.replace({'hateful': '0', 'abusive': '1', 'normal': '2'}).astype('int')

# concatinating
data = pd.concat([data_founta, data_davidson])
data = data[['tweet', 'count', 'label']]


# shuffling the dataset
data = data.sample(frac=1).reset_index(drop=True)


# creating an index and setting this as default 
data['id'] = data.index
data.set_index('id')
data.index.names = ['id']

# reordering the columns
data = data[['label', 'count', 'tweet']]

# saving the dataset
data.to_csv("coding/code/exchange_base/data.csv")