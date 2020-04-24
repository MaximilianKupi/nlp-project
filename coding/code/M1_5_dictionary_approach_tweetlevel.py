"""This is the script to define the function that searches for hatetful terms.
"""

# Loading packages

import torch
import pandas as pd
import numpy as np
import difflib

# Defining the function

def hatesearch(data = None, dictionary = None, verbose = False, average_hate = True, difflib_percentage = 0.85):
    """This function matches the terms in the Hatebase.org dictionary with the tweets in our dataset. Each word in the tweet gets assigned a number based on its hatefulness, based on hatebase.org (hatefulness is determined by the word appearing in the dictionary, its ambiguity as a term of hatespeech, its average offensiveness (as defined by hatebase.org methodology). The output is a tensor used for further analysis.
    
    Args:
        data (str): The input, a string (in this a tweet) from our dataframe. To be specified. Default: None.
        dictionary (dataframe): a dataframe containing hateful terms, default is an up-to-date extraction of English-language terms from Hatebase.org.
        verbose (bool): Whether or not outputs from print-Functions (used for testing purposes) should be shown
        average_hate (bool): Determines if a word in a tweet is found in multiple hate-dictionary entries, what value of hatefulness should be used. If "False", the maximum is used. If "True" (Default), the average is calculated.
        difflib_percentage (float): Percentage used to determine how sensible the matching function of words in the tweets with terms in the dictionary should be (in order to catch words with typos or small changes, which have been deliberately included in order to avoid detection, idea adapted from Chiu (2018): https://ethanchiu.xyz/blog/2018/02/03/Identifying-Hate/). Default: 0.85.

    Returns:
        Tensor of numbers for each word in the tweet, indicating hatefulness of the word
    """

    # Loading the data

    # loading Hatebase dictionary
    hatebase_path = 'coding/data/dictionary/hatebase/full_dictionary.csv'
    
    if dictionary is None:
        hatebase_dic = pd.read_csv(hatebase_path, index_col = 'vocabulary_id')
        #print('INFO: Reading dictionary anew from exchange_base since no input was provided')
    else:
        hatebase_dic = dictionary

    # lowercasing the hatebase dictionary
    hatebase_dic['term'] = hatebase_dic['term'].apply(lambda x: [y.lower() for y in x])
    hatebase_dic['term'] = hatebase_dic['term'].map(lambda x: "".join(x))

    averagevalue_of_offensiveness = hatebase_dic['average_offensiveness'].mean()

    # Splitting the tweets, storing each word in a list, and then a list of lists
    tweet_split = str(data).split()

    listoftweets = []
    for word in tweet_split:
        one_tweet = []
        one_tweet.append(word)
        listoftweets.append(one_tweet)

    # Searching through every word in the dictionary for every word in the tweet
    HateVector = []

    for word in listoftweets:
        final_hatefulness = 0
        if verbose:
            print(word)
        list_of_hate = []
        for hateterm in hatebase_dic['term']:
            frequency = 0
            hatefulness_term_weighted = 0
            offensiveness_value = 0
            Hatefulness = 0
            if hateterm in word or (len(difflib.get_close_matches(hateterm, word, 1, difflib_percentage)) == 1):
                frequency += 1
                offensiveness_value = hatebase_dic.loc[hatebase_dic['term'] == hateterm, 'average_offensiveness'].iloc[0]
                if verbose:
                    print("Single offensiveness value is", offensiveness_value)
                    print("Frequency", frequency)
                if np.isnan(offensiveness_value):
                    offensiveness_value = averagevalue_of_offensiveness
                else:
                    offensiveness_value = offensiveness_value
                if hatebase_dic.loc[hatebase_dic['term'] == hateterm, 'is_unambiguous'].iloc[0] == False:
                    hatefulness_term_weighted = offensiveness_value
                else:
                    hatefulness_term_weighted = offensiveness_value*2
                if verbose:
                    print("Weighted hatefulness value is", hatefulness_term_weighted)
                if frequency != 0:
                    Hatefulness = hatefulness_term_weighted
                else:
                    Hatefulness = 0    
            list_of_hate.append(Hatefulness)
            if verbose:
                print("List of hate numbers", list_of_hate)
        array_of_hate = np.array(list_of_hate)
        if not average_hate:
            final_hatefulness = np.amax(array_of_hate) 
        else:
            array_of_hate = array_of_hate[array_of_hate != 0]
            if verbose:
                print("Array of hate (without 0)", array_of_hate)   
            final_hatefulness = array_of_hate.mean()
            if np.isnan(final_hatefulness):
                final_hatefulness = 0
        HateVector.append(final_hatefulness)

    # Building a tensor from the vector created
    HateTensor = torch.tensor(HateVector)

    return HateTensor

# Testing on an example
#tweet = ("your wagon fish chief you retarded f*ggots")
#hatesearch(data = tweet, verbose = False)

