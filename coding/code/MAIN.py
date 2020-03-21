#### IMPORTING PACKAGES ####

## packages used in the MAIN file
import pandas as pd

## Packages used in our modules:

# packages used in M1_2_cleaning_data
    # import pandas as pd                       #   has to be installed
    # import re                                 #   normally perinstalled
    # import string                             #   normally preinstalled
    # import preprocessor as p                  #   has to be installed using pip
    # from nltk.tokenize import word_tokenize   #   has to installed
    # from nltk.corpus import stopwords         #   has to installed

# additional packages used in M1_3_splitting_datasets
    # from sklearn.model_selection import StratifiedShuffleSplit    # has to be installed

# additional packages used in M1_4_word_sentence_vectorisation
    # import transformers                       # has to be installed 
    # to install transformers rust has to be installed in the terminal beforehand with
        # curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        # Restart the terminal
        # pip install transformers==2.5.1


# additional packages used in M1_5_dictionary_approach       
    # import torch                              #   has to be installed
    # import spacy                              #   has to be installed
    # from pandarallel import pandarallel      #   has to be installed using pip               
    # language model for spacy has to be loaded using 'python -m spacy download en_core_web_sm' in the terminal



## loading our own created functions 

# function to clean the data
from M1_2_cleaning_data import data_cleaning
# function to split the data into train, val, and test set
from M1_3_splitting_datasets import split_data
# functino to vectorize the text data using pretrained BERT
from M1_4_word_sentence_vectorisation import vectorize
# function to search for hatebase dictionary terms in tweets
from M1_5_dictionary_approach import apply_dict



########################
##### OUR PIPELINE #####
########################

# getting the data from our exchange base
data = pd.read_csv("exchange_base/data.csv")

# cleaning the data
data_cleaned = data_cleaning(data=data)

# splitting the data 
train_set, val_set, test_set = split_data(data=data_cleaned)

# applying BERT vectorizer on train, validation and test set
# train set
train_matrix, train_labels = vectorize(train_set)
# val set
val_matrix, val_labels = vectorize(val_set)
# test set
test_matrix, test_labels = vectorize(test_set)

# applying dictionary approach
HateFrequency, HateIntensity, dataset_with_hatebasecount = apply_dict(data=data)

