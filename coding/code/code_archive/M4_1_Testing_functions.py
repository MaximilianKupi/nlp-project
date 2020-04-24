"""This is the script to test some of our functions
"""

### WORK IN PROGRESS

# %% 
# Importing necessary packages

import unittest
import pandas as pd
import numpy as np
import re
import string
import preprocessor as p # needs to be installed via pip and tweet-preprocessor as well
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import difflib
from M1_2_cleaning_data import clean_text
from M1_5_dictionary_approach_tweetlevel import hatesearch


class TestingFunctions(unittest.TestCase):

    def test_DataCleaning(self):
        original_tweet = "★THE GATEWAY 2★ ✔https://t.co/SSmqhC8rBA https://t.co/8jSwD7zC61 @Spokenamos #bookboost #ASMSG #kindle https://t.co/OdoRyxfrtp"
        function_cleaned_tweet = clean_text(text = original_tweet)
        ideal_cleaned_tweet = "the gateway "
        self.assertEqual(function_cleaned_tweet, ideal_cleaned_tweet)

    def test_HateSearch(self):
        input_tweet = "how could i be a fag but i like bitches please tell me"
        function_output = hatesearch(data = input_tweet)
        ideal_output = torch.tensor([ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 87.5849,  0.0000,  0.0000,
        85.0000, 25.0000,  0.0000,  0.0000,  0.0000])
        self.assertEqual(function_output, ideal_output)

if __name__ == '__main__':
    unittest.main()


# %%
from M1_2_cleaning_data import clean_text
original_tweet = "★THE GATEWAY 2★ ✔https://t.co/SSmqhC8rBA https://t.co/8jSwD7zC61 @Spokenamos #bookboost #ASMSG #kindle https://t.co/OdoRyxfrtp"
cleaned_text = clean_text(text = original_tweet)
ideal_cleaned_tweet = "the gateway "

print(ideal_cleaned_tweet)
print(cleaned_text)
ideal_cleaned_tweet == cleaned_text

# %%
