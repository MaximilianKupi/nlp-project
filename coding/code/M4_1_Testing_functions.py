"""This is the script to test the most important of our pre-processing functions.
"""


# Importing necessary packages

import unittest
import pandas as pd
import numpy as np
import random
import re
import string
import preprocessor as p # needs to be installed via pip and tweet-preprocessor as well
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import difflib
from M1_2_cleaning_data import *
from M1_5_dictionary_approach_tweetlevel import hatesearch
from M1_3_splitting_datasets import split_data
from M1_4_vectorisation_1d import vectorize as vectorize1D
from M1_4_vectorisation_2d import vectorize as vectorize2D

class TestingFunctions(unittest.TestCase):
    """Sets up various unittests.
    """
    
    def test_TextCleaning(self):
        """Tests the clean_text function from M1_2_cleaning_data based on one example tweet.
        """
        original_tweet = "★THE GATEWAY 2★ ✔https://t.co/SSmqhC8rBA https://t.co/8jSwD7zC61 @Spokenamos #bookboost #ASMSG #kindle https://t.co/OdoRyxfrtp"
        function_cleaned_tweet = clean_text(text = original_tweet)
        ideal_cleaned_tweet = "the gateway bookboost asmsg kindle"
        self.assertEqual(function_cleaned_tweet, ideal_cleaned_tweet)

    def test_TextCleaning_NoLowerCasing(self):
        """Tests the setter for lower_casing in the clean_text function from M1_2_cleaning_data based on one example tweet.
        """
        original_tweet = "★THE GATEWAY 2★ ✔https://t.co/SSmqhC8rBA https://t.co/8jSwD7zC61 @Spokenamos #bookboost #ASMSG #kindle https://t.co/OdoRyxfrtp"
        function_cleaned_tweet = clean_text(text = original_tweet, lower_casing=False)
        ideal_cleaned_tweet = "THE GATEWAY bookboost ASMSG kindle"
        self.assertEqual(function_cleaned_tweet, ideal_cleaned_tweet)

    def test_TextCleaning_NoDigitsRemoval(self):
        """Tests the setter for digits_removal in the clean_text function from M1_2_cleaning_data based on one example tweet.
        """
        original_tweet = "★THE GATEWAY 2★ ✔https://t.co/SSmqhC8rBA https://t.co/8jSwD7zC61 @Spokenamos #bookboost #ASMSG #kindle https://t.co/OdoRyxfrtp"
        function_cleaned_tweet = clean_text(text = original_tweet, digits_removal=False)
        ideal_cleaned_tweet = "the gateway 2 bookboost asmsg kindle"
        self.assertEqual(function_cleaned_tweet, ideal_cleaned_tweet)

    def test_TextCleaning_NoSymbolRemoval(self):
        """Tests the setter for punct_removal in the clean_text function from M1_2_cleaning_data based on one example tweet.
        """
        original_tweet = 'Jamice caption this picture *shows a picture from show* ""So u dressed like Sherlock Holmes but im embarrassing u?"" Lmaooooo love her"'
        function_cleaned_tweet = clean_text(text = original_tweet, symbol_removal=False)
        ideal_cleaned_tweet = 'jamice caption this picture *shows a picture from show* ""so u dressed like sherlock holmes but im embarrassing u?"" lmaooooo love her"'
        self.assertEqual(function_cleaned_tweet, ideal_cleaned_tweet)

    def test_DataCleaning(self):
        """Tests the text_column setter of the data_cleaning function from M1_2_cleaning_data works, based on an example dataframe.
        """
        original_df = pd.DataFrame({"text" : ["★THE GATEWAY 2★ ✔https://t.co/SSmqhC8rBA https://t.co/8jSwD7zC61 @Spokenamos #bookboost #ASMSG #kindle https://t.co/OdoRyxfrtp","Hey #Sweden how you liking those immigrants now? #Stockholm Wonder how many will say it's not radical islamic terro… https://t.co/7V9UWL3S5f"], "label":[2,0]})
        ideal_cleaned_df = pd.DataFrame({"text":["the gateway bookboost asmsg kindle", "hey sweden how you liking those immigrants now? stockholm wonder how many will say it's not radical islamic terro"],'label':[2,0]})
        function_cleaned_df = data_cleaning(data=original_df, text_column='text')
        self.assertTrue(function_cleaned_df.equals(ideal_cleaned_df))

    def test_SplitData(self):
        """Tests the exception error in the split_data function from M1_3_splitting_datasets in case percentages are set to sum up to more than one.
        """
        with self.assertRaises(Exception) as context:
            split_data(train_p=0.5, val_p=0.5, test_p=0.5)

        self.assertTrue('train_p + val_p + test_p should sum up to 1, however they sum up to: 1.5' in str(context.exception))

    def test_HateSearch(self):
        """Tests the hatesearch function from M1_5_dictionary_approach_tweetlevel creates correct tensors based on one example.
        """
        input_tweet = "how could i be a fag but i like bitches please tell me"
        function_output = hatesearch(data = input_tweet)
        ideal_output = torch.tensor([ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 87.60187224669603,  0.0000,  0.0000,
        85.0000, 25.0000,  0.0000,  0.0000,  0.0000])
        self.assertTrue(torch.equal(function_output, ideal_output))

    def test_VectoriserDimensions1D(self):
        """Tests the correct output dimensions of the vectorize function from M1_4_vectorisation_1d.
        """
        original_df = pd.DataFrame([["★THE GATEWAY 2★ ✔https://t.co/SSmqhC8rBA https://t.co/8jSwD7zC61 @Spokenamos #bookboost #ASMSG #kindle https://t.co/OdoRyxfrtp",2]], columns=['tweet', 'label'])
        function_embeddings, function_labels = vectorize1D(original_df)
        ideal_embeddings, ideal_labels = torch.tensor([np.random.randint(0, 9, 120)]), torch.tensor([2])  
        self.assertTrue(ideal_embeddings.size()==function_embeddings.size())
        self.assertTrue(function_labels.size() == ideal_labels.size())    

    def test_VectoriserDimensions2D(self):
        """Tests the correct output dimensions of the vectorize function from M1_4_vectorisation_2d.
        """
        original_df = pd.DataFrame([["★THE GATEWAY 2★ ✔https://t.co/SSmqhC8rBA https://t.co/8jSwD7zC61 @Spokenamos #bookboost #ASMSG #kindle https://t.co/OdoRyxfrtp",2]], columns=['tweet', 'label'])
        function_embeddings, function_labels = vectorize2D(original_df)
        ideal_embeddings, ideal_labels = torch.tensor([np.random.randint(0, 9, (2,120))]), torch.tensor([2])  
        self.assertTrue(ideal_embeddings.size()==function_embeddings.size())
        self.assertTrue(function_labels.size() == ideal_labels.size())


if __name__ == '__main__':
    unittest.main()





