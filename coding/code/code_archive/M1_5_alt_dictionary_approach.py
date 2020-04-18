# TODO create functions for different elements (apply) and overall function for main

# %% Loading packages

import pandas as pd
import numpy as np
import transformers
import torch
import tqdm
import spacy
from pandarallel import pandarallel

# %% Loading data and dictionary

# loading dictionary
hatebase_path = 'exchange_base/full_dictionary.csv'
hatebase_dic = pd.read_csv(hatebase_path, index_col = 'vocabulary_id')

# lemmatizing dictionary -- ### not necessary imho

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

# loading the data

data_path = "exchange_base/train_set.csv"
df = pd.read_csv(data_path, index_col = 'id')

# %% Splitting the tweets

tweet_i = []
for tweet in df['tweet']:
    tweet_split = str(tweet).split()
    tweet_i.append(tweet_split)
df['tweet_split'] = tweet_i
print(df.head)

# %% BERT vectorizer -- work in progress

# loading the model
pretrainedModel="bert-base-uncased"
tokenizer = transformers.BertTokenizer.from_pretrained(pretrainedModel)

# vectorizing an example tweet
tweet_example = df['tweet_split'][0]
print(tweet_example)
vectorized_tweet = tokenizer.encode(tweet_example)
print(vectorized_tweet)
TweetTensor = torch.tensor(vectorized_tweet)
print(TweetTensor)

# Result: output (15 numbers) > input (13 strings) -- why?

# vectorizing every tweet -- still problematic
'''
TweetVector = []
for word in df['tweet_split']:
    vectorized_tweet = tokenizer.encode(word)
    TweetVector.append(vectorized_tweet)
#print(TweetVector)
TweetTensor = torch.tensor(TweetVector)
#print(TweetTensor)
'''

# %% Apply dictionary on example (to show numbers in vector)

## Notes:
'''
    # a) Not lemmatizing, using just the terms (in lowercase), now can't find everyone word b/c lemmatization
    # b) Changing the way we look for words:
            # Now: for every term in the dictionary, check if it word from tweet 
                --> maximum accuracy, but not efficient
            # Alternative 1: check if word from tweet is in dictionary 
                --> efficient, but errors w/ small words & composites (e.g. "tard" in "retarded")
    # c) Problem: Currenty no way of finding hateterms in dictionary, made from two words, e.g. "fish wagon"
'''
samplevector = (["your", "wagon", "fish", "chief", "you", "retarded", "faggot"], ["you", "jewtarded", "fucking", "faggot", "you", "ass", "chief"])
HateVector_example = []

for tweet in samplevector:
    HateList=[]
    for word in tweet:
        frequency = 0
        hatefulness_term_weighted = 0
        offensiveness_value = 0
        Hatefulness = 0
        print(word)
        for hateterm in hatebase_dic['term_lemma']:
            frequency = 0
            hatefulness_term_weighted = 0
            offensiveness_value = 0
            if hateterm == word:
                frequency += 1
                offensiveness_value = hatebase_dic.loc[hatebase_dic['term_lemma'] == hateterm, 'average_offensiveness'].iloc[0]
                print("Single offensiveness value is", offensiveness_value)
                #print(frequency)
                if np.isnan(offensiveness_value):
                    offensiveness_value = 77.27734806629834
                else:
                    offensiveness_value = offensiveness_value
                if hatebase_dic.loc[hatebase_dic['term_lemma'] == hateterm, 'is_unambiguous'].iloc[0] == False:
                    hatefulness_term_weighted = offensiveness_value
                else:
                    hatefulness_term_weighted = offensiveness_value*2
                print("Weighted hatefulness value is", hatefulness_term_weighted)
                #print(frequency)
                if frequency != 0:
                    Hatefulness = hatefulness_term_weighted
                else:
                    Hatefulness = 0
        HateList.append(Hatefulness)
    HateVector_example.append(HateList)
print(HateVector_example)

HateTensor_example = torch.tensor(HateVector_example)
print(HateTensor_example)

#%% Apply dictionary on actual tweet (one)

HateList = []

for word in df['tweet_split'][0]:
    frequency = 0
    hatefulness_term_weighted = 0
    offensiveness_value = 0
    Hatefulness = 0
    print(word)
    for hateterm in hatebase_dic['term_lemma']:
        frequency = 0
        hatefulness_term_weighted = 0
        offensiveness_value = 0
        if hateterm == word:
            frequency += 1
            offensiveness_value = hatebase_dic.loc[hatebase_dic['term_lemma'] == hateterm, 'average_offensiveness'].iloc[0]
            print("Single offensiveness value is", offensiveness_value)
            #print(frequency)
            if np.isnan(offensiveness_value):
                offensiveness_value = 77.27734806629834
            else:
                offensiveness_value = offensiveness_value
            if hatebase_dic.loc[hatebase_dic['term_lemma'] == hateterm, 'is_unambiguous'].iloc[0] == False:
                hatefulness_term_weighted = offensiveness_value
            else:
                hatefulness_term_weighted = offensiveness_value*2
            print("Weighted hatefulness value is", hatefulness_term_weighted)
            #print(frequency)
            if frequency != 0:
                Hatefulness = hatefulness_term_weighted
            else:
                Hatefulness = 0
    HateList.append(Hatefulness)
print(HateList)

HateTensor = torch.tensor(HateList)
print(HateTensor)

# %% Merging BERT output and Hatebase output

matrix = torch.stack([TweetTensor, HateTensor])
print(matrix)