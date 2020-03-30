#### IMPORTING PACKAGES ####

## packages used in the MAIN file
import pandas as pd
from M2_0_NNSetup_EditsForTracking import *
from M2_1_CNN_1d import CNN_1d
from M2_1_CNN_1d_experiment import CNN_1d_experiment
import json

## Packages used in our modules:

# packages used in M1_2_cleaning_data
    # import pandas as pd                       #   has to be installed
    # import re                                 #   normally perinstalled
    # import string                             #   normally preinstalled
    # import preprocessor as p                  #   has to be installed using pip
    # from nltk.tokenize import word_tokenize   #   has to installed
    # from nltk.corpus import stopwords         #   has to installed

# additional packages used in M1_3_splitting_datasets
    # from sklearn.model_selection import StratifiedShuffleSplit    # has to be installed

# additional packages used in M1_4_word_sentence_vectorisation
    # import transformers                       # has to be installed 
    # to install transformers rust has to be installed in the terminal beforehand with
        # curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        # Restart the terminal
        # pip install transformers==2.5.1


# additional packages used in M1_5_dictionary_approach       
    # import torch                              #   has to be installed
    # import spacy                              #   has to be installed
    # from pandarallel import pandarallel       #   has to be installed using pip               
    # language model for spacy has to be loaded using 'python -m spacy download en_core_web_sm' in the terminal
    # import numpy as np                        #   normally preinstalled


## loading our own created functions 

# # function to clean the data
# from M1_2_cleaning_data import data_cleaning
# # function to split the data into train, val, and test set
# from M1_3_splitting_datasets import split_data
# # functino to vectorize the text data using pretrained BERT
# from M1_4_word_sentence_vectorisation import vectorize
# # function to search for hatebase dictionary terms in tweets
# from M1_5_dictionary_approach import apply_dict



########################
##### OUR PIPELINE #####
########################

# ## only done once at the beginning ##
# # getting the data from our exchange base
# data = pd.read_csv("exchange_base/data.csv")

# # cleaning the data
# data_cleaned = data_cleaning(data=data)

# # splitting the data 
# train_set, val_set, test_set = split_data(data=data_cleaned)


# ## reading the train, val, and test set
# train_set = pd.read_csv("exchange_base/train_set.csv")
# val_set = pd.read_csv("exchange_base/val_set.csv")
# test_set = pd.read_csv("exchange_base/test_set.csv")

# TODO vectorize three sets again with cleaned input and save as file

# applying BERT vectorizer on train, validation and test set
# train set
# TODO is it really necessary to do this ever time the application runs?
# train_matrix, train_labels = vectorize(train_set)
train_vectors = torch.load("coding/code/exchange_base/train_vectorized_1d.pt")
train_labels = torch.load("coding/code/exchange_base/train_labels_1d.pt")

# TODO is it really necessary to do this ever time the application runs?
# val set
# train_vectors, val_labels = vectorize(val_set)
val_vectors = torch.load("coding/code/exchange_base/val_vectorized_1d.pt")
val_labels = torch.load("coding/code/exchange_base/val_labels_1d.pt")

# TODO is it really necessary to do this ever time the application runs?
# test set
#train_vectors, test_labels = vectorize(test_set)

# TODO Question: Isn't that part of the preprocessing?
# applying dictionary approach
# HateFrequency, HateIntensity, dataset_with_hatebasecount = apply_dict(data=data)



# Config NN
# prefix to test different setups
uniqueInputPrefix = ""
uniqueOutputPrefix = "tracking_test_"
path = "coding/code/exchange_base/"
# Training input
stage = "train"
train_filpath_vectors = path + uniqueInputPrefix + stage +  "_vectorized.pt"
train_filepath_labels = path + uniqueInputPrefix + stage +  "_labels.pt"
# Model Training
epochs = 10
# Model Output
output_filepath_model = path + uniqueOutputPrefix + stage + "_model_epochs" + str(epochs) + ".ckpt"
# Evaluation
stage = "val"
val_filepath_vectors = path + uniqueInputPrefix + stage +  "_vectorized.pt"
val_filepath_labels = path + uniqueInputPrefix + stage +  "_labels.pt"
val_filepath_result_prefix = path + uniqueOutputPrefix + stage +  "_result.json"

variables =	{
    "global" : {
        "path" : path
    },
    "CNN" : {
        "layers" : {
            "1" : {
                "Conv1d" : {
                    "in_channels" : 120,
                    "out_channels" : 16,
                    "kernel_size" : 3
                },
                "BatchNorm1d" : {
                    "num_features" : 16
                },
                "MaxPool2d" : {
                    "kernel_size" : 2,
                    "stride" : 2
                }
            },
            "2" : {
                "Conv1d" : {
                    "in_channels" : 16,
                    "out_channels" : 32,
                    "kernel_size" : 3,
                },
                "BatchNorm1d" : {
                    "num_features" : 32
                },
                "MaxPool2d" : {
                    "kernel_size" : 2,
                    "stride" : 2
                }
            }
        },
        "fc.Linear" : {
            "in_features" : 32,
            "out_features" : 3
        }
    },
    "optimizer" : {
        "learning_rate" : 0.001
    },
    "training" : {
        "epochs" : epochs,
        "input" : {
            "batch_size": 16,
            "vectors": train_filpath_vectors, # only used with loadData function
            "labels": train_filepath_labels # only used with loadData function
        },
    },
    "output" : {
        "filepath" : output_filepath_model # only used when
    },
    "validation" : {
        "input" : {
            "model" : output_filepath_model,
            "result" : val_filepath_result_prefix,
            "batch_size": 1,
            "vectors": val_filepath_vectors,
            "labels": val_filepath_labels
        }
    }
}





# run NN

# This Main Python file uses these classes:
# - NNSetup class to setup the dataloading, the training and the evaluation    
# - CNN_1d_experiment class as the actual Neural Network
#
# If you want to edit the Neural network, edit it in the file: M2_1_CNN_1d_experiment.py


# potential adaptations to the model and training loop
class NNSetup_betterOptimizer(NNSetup):
    def __init__(self,variables):
         NNSetup.__init__(self,variables)

#     def setCriterion(self):
#         """ ??
#         """ 
#         self.criterion = nn.CrossEntropyLoss()



# Create new object of NNSetup class
setup = NNSetup_betterOptimizer(variables)

# load Data into the object NNSetup
setup.loadDataFromVariable("training",train_vectors,train_labels)
setup.loadDataFromVariable("validation",val_vectors,val_labels)

# Create Neural Network object from class nn.module
model = CNN_1d_experiment(variables)

# add model to NNSetup object
setup.addNN(model)

# define Criterion
setup.setCriterion()

# define Optimizer
setup.setOptimizer()

# run with demo limit
result = setup.train(demoLimit=5000, saveToFile=True) # result can be saved automatically with dictionary and train(self,saveToFile=True)

# run without demo limit
#result = setup.train(saveToFile=True) # result can be saved automatically with dictionary and train(self,saveToFile=True)

trainedModel = setup.getModel() # model can be saved automatically with dictionary and setup.saveModel()

# demo output
#print(json.dumps(result['epochs'], indent=2, sort_keys=True))