#### IMPORTING PACKAGES ####

## packages used in the MAIN file
import pandas as pd
import json
import platform

## loading our own created functions 

# for Preprocessing
# # function to clean the data
# from M1_2_cleaning_data import data_cleaning
# # function to split the data into train, val, and test set
# from M1_3_splitting_datasets import split_data
# # functino to vectorize the text data using pretrained BERT
# from M1_4_word_sentence_vectorisation import vectorize
# # function to search for hatebase dictionary terms in tweets
# from M1_5_dictionary_approach import apply_dict

# for Model Setup and Training
# # clsas to setup the 
from M2_0_NNSetup import *
from M2_1_CNN_1d import CNN_1d
from M2_1_CNN_1d_experiment import CNN_1d_experiment



########################
##### OUR PIPELINE #####
########################


#### PREPROCESSING ####

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


# applying BERT vectorizer on train, validation and test set
# train set

# train_matrix, train_labels = vectorize(train_set)

# val set
# train_vectors, val_labels = vectorize(val_set)

# test set
#train_vectors, test_labels = vectorize(test_set)

# applying dictionary approach
# HateFrequency, HateIntensity, dataset_with_hatebasecount = apply_dict(data=data)


#### MODEL AND TRAINING ####

# SETTING VARIABLES
variables =	{
    "global" : {
        "path" : "will_be_specified_based_on_plattform",
        "plattform": 'colab', # 'local' 'colab'
        "model_name" : "model"
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
        "type": "Adam", # "RMSprop", "Adam", "SGD"
        "learning_rate" : 0.0001,
        "momentum": 0.95
    },
    "training" : {
        "epochs" : 20,
        "sampler": True, # if sampler is set to false, shuffle will automatically be set to 'True' while training
        "softmax" : True,  
        "scheduler" : False, 
        "input" : {
            "batch_size": 16,
            "vectors": "will_get_specified", # only used with loadData function
            "labels": "will_get_specified" # only used with loadData function
        },
    },
    "output" : {
        "filepath" : "will_get_specified",
        "uniqueOutputPrefix": "will_get_specified" # only used when
    },
    "validation" : {
        "input" : {
            "model" : "will_get_specified",
            "result" : "will_get_specified",
            "batch_size": 1,
            "vectors": "will_get_specified",
            "labels": "will_get_specified"
        }
    }
}


# specify prefixes and paths based on variables in dictionary 
uniqueInputPrefix = ""
uniqueOutputPrefix = variables['general']['model'] + "_optimizer_" + variables['optimzer']['type'] + "_lr_" + variables['optimzer']['learning_rate'] + "_epochs_" + variables['training']['epochs'] + "_batchsize_" + variables['training']['input']['batch_size'] + "_sampler_" + variables['training']['sampler'] + "_softmax_" + variables['training']['softmax'] + "_scheduler_" + variables['training']['scheduler'] 


if variables['global']['platform'] == 'colab':
    variables['global']['path'] = "exchange_base/" 
elif variables['global']['platform'] == 'local':
    variables['global']['path'] = "coding/code/exchange_base/"
else:
    print('Please specify platform in variables')
    raise ValueError

# Training input
stage = "train"
variables['training']['input']['vectors'] = variables['global']['path'] + uniqueInputPrefix + stage +  "_vectorized.pt"
variables['training']['input']['labels'] = variables['global']['path']+ uniqueInputPrefix + stage +  "_labels.pt"
# Model Output
save_dir = variables['global']['path'] + "Model_Results/" + variables["output"]["uniqueOutputPrefix"]
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
variables['output']['filepath'] = save_dir
# Evaluation
stage = "val"
variables['validation']['input']['vectors'] = variables['global']['path']+ uniqueInputPrefix + stage +  "_vectorized.pt"
variables['validation']['input']['labels'] = variables['global']['path']+ uniqueInputPrefix + stage +  "_labels.pt"
variables['validation']['input']['results'] = variables['global']['path'] + uniqueOutputPrefix + "_" + stage +  "_result.json"


# loading the data saved during preprocessing: 
train_vectors = torch.load(variables['global']['path'] + "train_vectorized_1d.pt")
train_labels = torch.load(variables['global']['path'] + "train_labels_1d.pt")

# val set
# train_vectors, val_labels = vectorize(val_set)
val_vectors = torch.load(variables['global']['path'] + "val_vectorized_1d.pt")
val_labels = torch.load(variables['global']['path'] + "val_labels_1d.pt")


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

# Create Neural Network object with the model from class
#model = CNN_1d_experiment(initial_num_channels=1, num_channels=256, hidden_dim=256, num_classes=3, dropout_p=0.1)

# add model to NNSetup object
setup.addNN(model)

# define Criterion
setup.setCriterion()

# define Optimizer
setup.setOptimizer()

# set the Scheduler
setup.setScheduler()

# run with demo limit
#result = setup.train(demoLimit=5000, saveToFile=True) # result can be saved automatically with dictionary and train(self,saveToFile=True)

# run without demo limit
result = setup.train(saveToFile=True) # result can be saved automatically with dictionary and train(self,saveToFile=True)

trainedModel = setup.getModel() # model can be saved automatically with dictionary and setup.saveModel()

# demo output
#print(json.dumps(result['epochs'], indent=2, sort_keys=True))