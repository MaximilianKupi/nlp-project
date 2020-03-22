# Importing the required modules

import pandas as pd
import re
import string
import preprocessor as p
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def data_cleaning(data = None, standard_twitter_cleaning = True, lower_casing = True, digits_removal = True, punct_removal = True, whitespace_removal = True):
    """This funtion cleans the dataset by performing standard Twitter cleaning (removing URLs, mentions, hashtags, reserved words, emojis and smileys), changing the text to lower case, and removing punctuation, whitespaces and standalone digits.
        The input dataframe can be specified with the argument "data". If nothing is specified, the function will get the data from our GitHub repository.
        The data cleaning can be customized by setting the respective methods to False:
    standard_twitter_cleaning = True, lower_casing = True, digits_removal = True, punct_removal = True, whitespace_removal = True"""
    
    # Setting the exchange_base link from where to get the data

    merged_data_path = "coding/code/exchange_base/data.csv"

    # Reading the data in case it is not here already (since it will later all be applied in one pipeline)
    
    if data is None:
        data = pd.read_csv(merged_data_path, index_col = "id")
        print('INFO: Reading data anew from exchange_base since no input was provided')
    
    # Creating the cleaning function

    def clean_data(text):
        
        # Removing emojis, smileys, URLs, mentions, hashtags and reserved words

        if standard_twitter_cleaning:
            p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)
            new_text = p.clean(text)
        
        # Converting to lower case

        if lower_casing:
            new_text = new_text.lower()

        # Removing standalone digits
        
        if digits_removal:
            new_text = re.sub(r"\b\d+\b", "", new_text)

        # Removing punctuation

        if punct_removal:
            new_text = re.sub(r"[^\w\d'\s]+","",new_text)

        # Removing double spaces and whitespaces
        
        if whitespace_removal:
            new_text = re.sub('\s+', ' ', new_text)
            new_text.strip()
        
        return new_text

    # Applying the function to the whole tweets column and creating the cleaned dataset

    data['tweet'] = data['tweet'].apply(lambda x : clean_data(x))
    data_cleaned = data

    return data_cleaned



# part to run only if the py file is run directly
if __name__ == "__main__":
    data_cleaned = data_cleaning()


    # output_file_name = "exchange_base/data_cleaned.csv"
    # 2. use exchange_base files
    path = "coding/code/exchange_base/"
    output_file_path = path +  "data_cleaned.csv"

    # Saving the cleaned dataset
    data_cleaned.to_csv(output_file_path)
