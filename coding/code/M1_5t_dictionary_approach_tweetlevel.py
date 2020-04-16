# %% Loading packages

import torch
import pandas as pd
import numpy as np

# Defining the function

def hatesearch(data = None, dictionary = None):
    """This function matches the terms in the Hatebase.org dictionary with the tweets in our dataset.
    Each word in the tweet gets assigned a number based on its hatefulness, based on hatebase.org
    (hatefulness is determined by the word appearing in the dictionary, its ambiguity as a term of hatespeech, 
    its average offensiveness (as defined by hatebase.org methodology). 
    The output is a tensor used for further analysis.
    The input dataframe (tweets) needs to be specified with the variable "data". """

    # Loading the data

    # loading Hatebase dictionary
    hatebase_path = "exchange_base/full_dictionary.csv"
    
    if dictionary is None:
        hatebase_dic = pd.read_csv(hatebase_path, index_col = 'vocabulary_id')
        print('INFO: Reading dictionary anew from exchange_base since no input was provided')
    else:
        hatebase_dic = dictionary

    # lowercasing the hatebase dictionary
    hatebase_dic['term'] = hatebase_dic['term'].apply(lambda x: [y.lower() for y in x])
    hatebase_dic['term'] = hatebase_dic['term'].map(lambda x: "".join(x))

    # Data specifier reminder
    if data is None:
        print('Please specify the data input')

    # Splitting the tweets
    tweet_split = str(data).split()

    # Searching through every word in the dictionary for every word in the tweet
    HateVector = []

    for word in tweet_split:
        frequency = 0
        hatefulness_term_weighted = 0
        offensiveness_value = 0
        Hatefulness = 0
        print(word)
        for hateterm in hatebase_dic['term']:
            frequency = 0
            hatefulness_term_weighted = 0
            offensiveness_value = 0
            if hateterm == word:
                frequency += 1
                offensiveness_value = hatebase_dic.loc[hatebase_dic['term'] == hateterm, 'average_offensiveness'].iloc[0]
                print("Single offensiveness value is", offensiveness_value)
                #print(frequency)
                if np.isnan(offensiveness_value):
                    offensiveness_value = 77.27734806629834
                else:
                    offensiveness_value = offensiveness_value
                if hatebase_dic.loc[hatebase_dic['term'] == hateterm, 'is_unambiguous'].iloc[0] == False:
                    hatefulness_term_weighted = offensiveness_value
                else:
                    hatefulness_term_weighted = offensiveness_value*2
                print("Weighted hatefulness value is", hatefulness_term_weighted)
                #print(frequency)
                if frequency != 0:
                    Hatefulness = hatefulness_term_weighted
                else:
                    Hatefulness = 0
        HateVector.append(Hatefulness)
    print(HateVector)
    #return HateVector

    # Building a tensor from the vector created
    HateTensor = torch.tensor(HateVector)

    return HateTensor

# Testing on an example
#tweet = ("your wagon fish chief you retarded faggot")
#hatesearch(data = tweet)

# %%
