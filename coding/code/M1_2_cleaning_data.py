"""This is the script to defined the data cleaning function for our preprocessing.
"""

# Importing the required packages
import pandas as pd
import re
import string
import preprocessor as p
# TODO make sure pip requirements are documented
# pip3 install preprocessor is not enoug for this to work
# pip3 install tweet-preprocessor is also necessary, although it doesn't have to be imported separately
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Defining the cleaning function

def clean_text(text, standard_twitter_cleaning = True, lower_casing = True, digits_removal = True, punct_removal = True, whitespace_removal = True):
    """Cleans any string of text by performing standard Twitter cleaning (removing URLs, mentions, hashtags, reserved words, emojis and smileys), changing the text to lower case, and removing punctuation, whitespaces and standalone digits.

    Args:
        text (str): The input text to clean.
      
        
        standard_twitter_cleaning (bool): Whether or not to use standard twitter cleaining package. Default: True. 
        lower_casing (bool): Whether or not to lower case the words. Default: True.
        digits_removal (bool): Whether or not to remove digits. Default: True.
        punct_removal (bool): Whether or not to remove anything but words. Default: True.
        whitespace_removal (bool): Whether or not to remove excessive white space in between words. Default: True.

    Returns:
        The cleaned text.
    """
    
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
    
    cleaned_text = new_text

    return cleaned_text


# defining the function to go thru the dataframe
def data_cleaning(data = None, text_column ='tweet', standard_twitter_cleaning = True, lower_casing = True, digits_removal = True, punct_removal = True, whitespace_removal = True):
    """Applies the clean_text function to a dataframe where the tweets are saved as strings in a column. 
    
    Args:
        data (pandas dataframe): Specify the input dataframe. If nothing is specified, the function will get the data from our exchange base folder.
        text_column (str): Specify the column where the tweets are stored as strings. Default: 'tweet'.
        standard_twitter_cleaning (bool): Whether or not to use standard twitter cleaining package. Default: True. (will be passed on to the clean_text function)
        lower_casing (bool): Whether or not to lower case the words. Default: True. (will be passed on to the clean_text function)
        digits_removal (bool): Whether or not to remove digits. Default: True. (will be passed on to the clean_text function)
        punct_removal (bool): Whether or not to remove anything but words. Default: True. (will be passed on to the clean_text function)
        whitespace_removal (bool): Whether or not to remove excessive white space in between words. Default: True. (will be passed on to the clean_text function)

    Returns:
        The cleaned data frame.
    """

    # Setting the exchange_base link from where to get the data

    merged_data_path = "coding/code/exchange_base/data.csv"

    # Reading the data in case it is not here already (since it will later all be applied in one pipeline)
    
    if data is None:
        data = pd.read_csv(merged_data_path, index_col = "id")
        print('INFO: Reading data anew from exchange_base since no input was provided')
    

    # Applying the function to the whole tweets column and creating the cleaned dataset
    data[text_column] = data[text_column].apply(lambda x : clean_text(x, standard_twitter_cleaning, lower_casing, digits_removal, punct_removal, whitespace_removal))
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
