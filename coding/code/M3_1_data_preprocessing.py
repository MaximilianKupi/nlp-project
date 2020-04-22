"""The main file to run the data pre processing pipeline.
"""
##########################
##### PRE PROCESSING #####
##########################

#### IMPORTING PACKAGES ####

# general functions
import pandas as pd

## loading our own created functions 

# for Preprocessing

# function to clean the data
from M1_2_cleaning_data import data_cleaning
# function to split the data into train, val, and test set
from M1_3_splitting_datasets import split_data
# functino to vectorize the text data using pretrained BERT
from M1_4_vectorisation_1d import vectorize,createTensors
from M1_4_vectorisation_2d import vectorize,createMatrix,padWithZeros,createTensors,stretch
# function to search for hatebase dictionary terms in tweets
from M1_5_dictionary_approach import apply_dict

#### PREPROCESSING ####

# to skip all following code when documenting with sphynx
if __name__ == "__main__":

    path = "coding/code/exchange_base/"

    ## only done once at the beginning ##
    # getting the data from our exchange base
    data = pd.read_csv(path + "data.csv")

    ### cleaning the data ###
    data_cleaned = data_cleaning()
    # output_file_name = "exchange_base/data_cleaned.csv"
    # 2. use exchange_base files
    
    output_file_path = path +  "data_cleaned.csv"
    # Saving the cleaned dataset
    data_cleaned.to_csv(output_file_path)

    # splitting the data 
    train_set, val_set, test_set = split_data(data=data_cleaned)
    # saving the dataframes
    train_set.to_csv("coding/code/exchange_base/train_set.csv")
    val_set.to_csv("coding/code/exchange_base/val_set.csv")
    test_set.to_csv("coding/code/exchange_base/test_set.csv")


    # TODO: redo the parts for the vectorization and dictionary approach

    # applying BERT vectorizer on train, validation and test set
    path = "coding/code/exchange_base/"

    createTensors(path,"train")
    createTensors(path,"val")
    createTensors(path,"test")

    # applying dictionary approach
    HateFrequency, HateIntensity, dataset_with_hatebasecount = apply_dict(data=data)