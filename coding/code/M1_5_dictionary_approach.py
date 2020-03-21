# Loading packages

import pandas as pd
import torch
import spacy
from pandarallel import pandarallel

def apply_dict(data = None):
    """This function matches the terms in the Hatebase.org dictionary with the tweets in our dataset.
    First, both datasets are lemmatized, then the function counts the number of hatebase.org terms appearing in each tweet and adds the count as well as the average offensiveness (as defined by hatebase.org methodology). These two outputs are then added to the original dataset and further transformed into a tensor for further analysis.
    The input dataframe (tweets) can be specified with the variable "data". If nothing is specified, the function will get the data from our GitHub repository."""

    # Loading the data

    # loading Hatebase dictionary
    hatebase_url = "https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/data/dictionary/hatebase/full_dictionary.csv"
    hatebase_dic = pd.read_csv(hatebase_url, index_col = 'vocabulary_id')

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
    pandarallel.initialize() 

    # defining the search function
    def hatesearch(row):
        frequency = 0
        hatefulness = 0
        for term_lemma in hatebase_dic['term_lemma']:
            if term_lemma in row['tweet_lemma']:
                frequency += 1
                hatefulness = hatebase_dic[hatebase_dic['term_lemma'] == term_lemma].average_offensiveness
        row['Hatefreq'] = frequency
        row['Hatefulness'] = hatefulness
        return row


    # running the search function
    data = data.parallel_apply(hatesearch, axis = 1)

    # Building a tensor out of the additional columns

    HateFrequency = torch.tensor(data['Hatefreq'])

    HateIntensity = torch.tensor(data['Hatefulness'])

    dataset_with_hatebasecount = data

    return HateFrequency, HateIntensity, dataset_with_hatebasecount


if __name__ == "__main__":
    # Running the function

    HateFrequency, HateIntensity = apply_dict()

    # output_file_name = "exchange_base/hatefreq.pt"
    # 2. use exchange_base files
    path = "exchange_base/"

    # input_file = path + stage + "_set.csv"
    output_file_name_hatefreq = path +  "hatefreq.pt"
    output_file_name_hateint = path + "hateint.pt"

    ### Saving the data
    # Saving the vectorized tweets tensor
    torch.save(HateFrequency, output_file_name_hatefreq)
    # Saving the label tensor
    torch.save(HateIntensity, output_file_name_hateint)
