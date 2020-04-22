"""Merging and Shuffling the Datasets

This is the script which has to be run at the beginning to merge and shuffle the datasets of Founta et al. and Davidson et al.
"""

# loading packages
import pandas as pd
import numpy as np

# setting seed for reproducability
np.random.seed(42)


# prevent file from being run while documenting
if __name__ == "__main__":
    # loading dataframes and adapting labels and headers

    # setting the paths where to get the data from exchange_base
    path_davidson = "../data/twitter data/davidson et al/labeled_data.csv"
    path_founta = "../data/twitter data/founta et al/hatespeech_text_label_vote.csv"


    data_davidson = pd.read_csv(path_davidson, names=["count", "hate_speech", "offensive_language", "neither", "label", "tweet"], header=1)
    data_founta = pd.read_csv(path_founta, sep='\t', names=["tweet", "label_text", "count"])

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
    data.to_csv("./exchange_base/data.csv")