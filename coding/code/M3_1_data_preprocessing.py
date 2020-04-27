"""The main file to run the data preprocessing pipeline.
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

from M1_4_vectorisation_1d import createTensors as createTensors_1D

from M1_4_vectorisation_2d import createTensors as createTensors_2D

#### PREPROCESSING ####

# to skip all following code when documenting with sphynx
if __name__ == "__main__":

    path = "coding/code/exchange_base/"

    ## only done once at the beginning ##
    # getting the data from our exchange base
    data = pd.read_csv(path + "data.csv")

    ### cleaning the data ###
    data_cleaned = data_cleaning()
    
    output_file_path = path +  "data_cleaned.csv"
    
    # Saving the cleaned dataset
    data_cleaned.to_csv(output_file_path)

    # splitting the data 
    train_set, val_set, test_set = split_data(data=data_cleaned)

    # saving the dataframes
    train_set.to_csv(path + "train_set.csv")
    val_set.to_csv(path + "val_set.csv")
    test_set.to_csv(path + "test_set.csv")


    # applying BERT vectorizer on train, validation and test set

    # for the 1D model
    createTensors_1D(path,"train")
    createTensors_1D(path,"val")
    createTensors_1D(path,"test")

    # for the 2D model plus dictionary approach
    createTensors_2D(path,"train")
    createTensors_2D(path,"val")
    createTensors_2D(path,"test")
