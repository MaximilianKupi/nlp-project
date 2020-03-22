# Loading packages

import pandas as pd
import torch
import spacy
from pandarallel import pandarallel
import numpy as np 

def apply_dict(data = None, dictionary = None):
    """This function matches the terms in the Hatebase.org dictionary with the tweets in our dataset.
    First, both datasets are lemmatized, then the function counts the number of hatebase.org terms appearing in each tweet and adds the count as well as the average offensiveness (as defined by hatebase.org methodology). These two outputs are then added to the original dataset and further transformed into a tensor for further analysis.
    The input dataframe (tweets) can be specified with the variable "data". If nothing is specified, the function will get the data from our GitHub repository."""

    # Loading the data

    # loading Hatebase dictionary
    hatebase_url = "https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/data/dictionary/hatebase/full_dictionary.csv"
    
    if dictionary is None:
        hatebase_dic = pd.read_csv(hatebase_url, index_col = 'vocabulary_id')
        print('INFO: Reading dictionary anew from GitHub since no input was provided')
    else:
        hatebase_dic = dictionary

    # github url for loading the tweets in case they are not provided
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
        
    #joining list of lemmas into one string again, so it becomes matchable by the searchfunction
    hatebase_dic['term_lemma'] = hatebase_dic['term_lemma'].map(lambda lemmas: " ".join(lemmas))

    hatebase_dic = hatebase_dic.drop_duplicates(subset=['term_lemma'])

    # lemmatizing the tweets
    lemma = []

    for tweet in nlp.pipe(data['tweet'].astype('unicode').values, batch_size=80, n_threads=5):
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
    
    # defining the search function
    def hatesearch(row):
        #print(row)
        frequency = 0
        hatefulness_sum = 0
        for hateterm in hatebase_dic['term_lemma']:
            if hateterm in row['tweet_lemma']:
                frequency += 1
                row_hatebase_dict_term_lemma = hatebase_dic.loc[hatebase_dic['term_lemma'] == hateterm, 'average_offensiveness'].iloc[0]
                if np.isnan(row_hatebase_dict_term_lemma):
                    hatefulness_term = 77.27734806629834
                else: 
                    hatefulness_term = row_hatebase_dict_term_lemma
                hatefulness_sum += hatefulness_term
        row['Hatefreq'] = frequency
        row['Hatefulness'] = (hatefulness_sum / frequency)
        return row


    # running the search function
    pandarallel.initialize() 
    data = data.parallel_apply(hatesearch, axis = 1)
    #data = data.apply(hatesearch, axis = 1)

    # Building a tensor out of the additional columns

    HateFrequency = torch.tensor(data['Hatefreq'].values.astype("float16"))

    HateIntensity = torch.tensor(data['Hatefulness'].values.astype("float16"))

    dataset_with_hatebasecount = data

    return HateFrequency, HateIntensity, dataset_with_hatebasecount


if __name__ == "__main__":
    # Running the function

    HateFrequency, HateIntensity, dataset_with_hatebasecount = apply_dict()

    # output_file_name = "exchange_base/hatefreq.pt"
    # 2. use exchange_base files
    path = "coding/code/exchange_base/"

    # specifying output filenames
    output_file_name_hatefreq = path +  "hatefreq.pt"
    output_file_name_hateint = path + "hateint.pt"
    output_file_name_dataset_with_hatebasecount = path + "dataset_with_hatebasecount.csv"
    
    ### Saving the data
    # Saving the vectorized tweets tensor
    torch.save(HateFrequency, output_file_name_hatefreq)
    # Saving the label tensor
    torch.save(HateIntensity, output_file_name_hateint)
    # saving the augmented dataset
    dataset_with_hatebasecount.to_csv(output_file_name_dataset_with_hatebasecount)