# Loading packages
#import csv
#import sys
#import nltk
import pandas as pd
import torch
import spacy
import dframcy
import pandarallel

def dict_approach(data = None)
    """This function matches the terms in the Hatebase.org dictionary with the tweets in our dataset.
    First, both datasets are lemmatized, then the function counts the number of hatebase.org terms appearing in each tweet and adds the count as well as the average offensiveness (as defined by hatebase.org methodology). These two outputs are then added to the original dataset and further transformed into a tensor for further analysis.
    The input dataframe (tweets) can be specified with the variable "data". If nothing is specified, the function will get the data from our GitHub repository."""

    # Loading the data

    # loading Hatebase dictionary
    hatebase_url = "https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/data/dictionary/hatebase/full_dictionary.csv"
    hatebase_dic = pd.read_csv(hatebase_url, index_col = 'vocabulary_id')

    # loading the tweets
    data_url = "https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/code/exchange_base/data.csv"

    # reading the data in case it is not here already (since it will later all be applied in one pipeline)
    if data is None:
        data = pd.read_csv(data_url, index_col = 'id')
        print('INFO: Reading data anew from GitHub since no input was provided')

    # Lemmatizing both the dictionary and the tweets
    
    # loading the spacy model
    nlp = spacy.load('en_core_web_sm')
    # switching off irrelevant spacy functions
    nlp.disable_pipes('tagger', 'ner')

    # lemmatizing the dictionary
    lemma = []
    for term in nlp.pipe(hatebase_dic['term'].astype('unicode').values, batch_size=50, n_threads=3):
        if term.is_parsed:
            lemma.append([n.lemma_ for n in term])
        else:
            lemma.append(None)

    # storing the lemmas in a new column 
    hatebase_dic['term_lemma'] = lemma

    #making lemmas lowercase
    hatebase_dic['term_lemma'] = hatebase_dic['term_lemma'].map(lambda lemmas: [x.lower() for x in lemmas])
        
    #joining list of lemmas into one string agein, so it becomes matchable by the searchfunction
    hatebase_dic['term_lemma'] = hatebase_dic['term_lemma'].map(lambda lemmas: " ".join(lemmas))

    # lemmatizing the tweets
    lemma = []
    for tweet in nlp.pipe(data['tweet'].astype('unicode').values, batch_size=50, n_threads=3):
        if tweet.is_parsed:
            lemma.append([n.lemma_ for n in tweet])
        else:
            lemma.append(None)

    # storing the lemmas in a new column 
    data['tweet_lemma'] = lemma

    #making lemmas lowercase
    data['tweet_lemma'] = data['tweet_lemma'].map(lambda lemmas: [x.lower() for x in lemmas])
        
    #joining list of lemmas into one string agein, so it becomes matchable by the searchfunction
    data['tweet_lemma'] = data['tweet_lemma'].map(lambda lemmas: " ".join(lemmas))

    # Searching for HateBase words in tweets

    pandarallel.initialize() 

    # defining the search function
    def hatesearch(data):
        frequency = 0
        hatefulness = 0
        for term_lemma in hatebase_dic:
            if term_lemma in data['tweet_lemma']:
                frequency += 1
                hatefulness = hatebase_dic[hatebase_dic['term_lemma'] == term_lemma].average_offensiveness
        data['Hatefreq'] = frequency
        data['Hatefulness'] = hatefulness
        return data

    # running the search function
    data = data.parallel_apply(hatesearch, axis = 1)

    # Building a tensor out of the additional columns

    HateFrequency = torch.tensor(data['Hatefreq'])
    torch.save(HateFrequency, 'https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/code/exchange_base/HateFreq_Tensor.pt')

    HateIntensity = torch.tensor(data['Hatefulness'])
    torch.save(HateFrequency, 'https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/code/exchange_base/HateIntens_Tensor.pt')