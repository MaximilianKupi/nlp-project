"""The main file to run the testing of the model.
"""

#########################
######## TESTING ########
#########################

#### IMPORTING PACKAGES ####

## packages used in the MAIN file
import pandas as pd
import json
import platform
from sklearn.model_selection import ParameterGrid
import csv 

## loading our own created functions 

# for Model Setup and Training

# # class to setup the model
#from M2_1_CNN_1d import CNN_1d
# since we are still in the experimenting mode we use the CNN experiment
from M2_1_CNN_1d_experiment import CNN_1d_experiment
from M2_1_CNN_2d_experiment import CNN_2d_experiment


# # class to setup the dataloading, the training and the evaluation  
from M2_0_NN_setup import *

# keeping script from running while documenting
if __name__ == "__main__":
    # SETTING VARIABLES
    path_of_the_model_to_test = "/Users/mxm/Google Drive/Masterstudium/Inhalte/4th Semester/NLP/nlp-project/coding/code/exchange_base/Model_Results/Training_26_27_28_2D_60_epochs/26_CNN_experiment_2D_optimizer_Adam_lr_01_epochs_60_batchsize_16_samplerTclassweightsF_False_scheduler_True/Model_epoch_35.pt"

    variables =	{
        "global" : {
            "platform": 'local', # 'local' 'colab'
            "model_name" : "CNN_experiment_2D", 
            "grid_search_name" : "final_2D_gs26",
            "dimension_of_model" : "2D" #2D
        },
        "optimizer" : {
            "type": ["Adam", "RMSprop",  "SGD"],
            "learning_rate" : [0.0001, 0.001, 0.01],
            "momentum": 0.9
        },
        "training" : {
            "epochs" : 90,
            "sampler_true_class_weights_false": [True,False], # If set to True Sampler is set to True and Class Weights of Loss Function Criterion are set to False 
            #If set to False Sampler is set to False and Class Weights are set True. In this case shuffle will be automatically set to True during training. 
            "scheduler" : [True, False], 
            "input" : {
                "batch_size": 16,
            },
        },
        "validation":{
            "input" : {
                "batch_size": 1,
            },
            "output" : {
            #will be filled on the go
            },
        },
        "testing":{
            "input" : {
                "batch_size": 1,
            },
            "output" : {
            #will be filled on the go
            },
        },
        "output": {
            "results" : {
            #will be filled on the go
            },
        },
    }

    # SPECIFYING  PREFIXES AND FILEPATHS 

    uniqueInputPrefix = ""

    # specifying uniqueOutputPrefix based on the used parameters
    uniqueOutputPrefix = "testing_" + variables['global']['model_name']

    # setting general path
    if variables['global']['platform'] == 'colab':
        variables['global']['path'] = "exchange_base/" 
    elif variables['global']['platform'] == 'local':
        variables['global']['path'] = "/Users/mxm/Google Drive/Masterstudium/Inhalte/4th Semester/NLP/nlp-project/coding/code/exchange_base/"
    else:
        print('Please specify platform in variables')
        raise ValueError


    # Specifying directory to save output
    save_dir = variables['global']['path'] + "Model_Results/Testing/" + variables['global']['grid_search_name'] + "/" + uniqueOutputPrefix
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    variables['output']['filepath'] = save_dir

    # Path to save the dictionary of whole model performance per epoch of run
    results_json_path = variables['output']['filepath'] +   "/all_results_of_model.json"
    variables['testing']['output']['results'] = results_json_path

    # loading the vectors and labels from the exchange base in 1d or 2d
    if variables['global']['dimension_of_model'] == '1D': 
        # loading the data saved during preprocessing: 
        test_vectors = torch.load(variables['global']['path'] + "test_vectorized_1d.pt")
        test_labels = torch.load(variables['global']['path'] + "test_labels_1d.pt")

    elif variables['global']['dimension_of_model'] == '2D': 
        # loading the data saved during preprocessing: 
        test_vectors = torch.load(variables['global']['path'] + "test_vectorized_2d.pt")
        test_labels = torch.load(variables['global']['path'] + "test_labels_2d.pt")

    else:
        print('Please specify correct dimension of model.')

    # RUNNING THE MODEL

    # Create new object of NNSetup class
    setup = NN_Training_Setup(variables)

    # load Data into the object NNSetup
    setup.loadDataFromVariable("testing",test_vectors,test_labels)

    # Create Neural Network object based on the modules CNN_1d_experiment (if 1D) or CNN_2d_experiment (if 2D)
    if variables['global']['dimension_of_model'] == '1D': 
        model = CNN_1d_experiment(variables)

    elif variables['global']['dimension_of_model'] == '2D': 
        model = CNN_2d_experiment(variables)

    else:
        print('Please specify correct dimension of model.')


    # add model to NNSetup object
    setup.addNN(model)

    # loading the state dictionary of the model
    setup.loadModel(path_of_the_model_to_test)

    # testing the model and saving the results
    setup.testModel()