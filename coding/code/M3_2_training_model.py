"""The main file to run the training of model.
"""

########################
####### TRAINING #######
########################

#### IMPORTING PACKAGES ####

## packages used in the MAIN file
import pandas as pd
import json
import platform
import sys
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
from M2_0_NN_training_setup import *


# SETTING VARIABLES
variables =	{
    "global" : {
        "platform": 'colab', # 'local' 'colab'
        "model_name" : "CNN_experiment_1D",
        "grid_search_name" : 'SecondGridSearch_1D_withPadding_and_Seed', #"SecondGridSearch_1D_withPadding_and_seed",
        "dimension_of_model" : "1D", #2D,
        "seed" : 42
    },
    "optimizer" : {
        "type": ["Adam", "RMSprop",  "SGD"],
        "learning_rate" : [0.0001, 0.001, 0.01],
        "momentum": 0.9
    },
    "training" : {
        "epochs" : 40,
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
    "output": {
        "results" : {
        #will be filled on the go
        },
    },
}

# SETTING-UP HYPER PARAMETER GRID SEARCH
param_grid = {  'optimizer_type':variables['optimizer']['type'], 
                'learning_rate': variables['optimizer']['learning_rate'], 
                "sampler_true_class_weights_false" : variables['training']["sampler_true_class_weights_false"],
                "scheduler" : variables['training']["scheduler"]}

all_params = list(ParameterGrid(param_grid))


# to skip all following code when documenting with sphynx
if __name__ == "__main__":

    # Running through the grid
    for run_number, current_params in enumerate(all_params):
        
        if run_number == 1000:
            print('skipping', run_number)
        else:
            # printing current parameters
            print(current_params)

            # writing the parameters of that grid search run into the dictionary
            variables['optimizer']['learning_rate'] = current_params['learning_rate']
            variables['optimizer']['type'] = current_params['optimizer_type']
            variables['training']["sampler_true_class_weights_false"] = current_params["sampler_true_class_weights_false" ]
            variables['training']["scheduler"] = current_params['scheduler']


            # SPECIFYING  PREFIXES AND FILEPATHS 

            uniqueInputPrefix = ""

            # specifying uniqueOutputPrefix based on the used parameters
            uniqueOutputPrefix = str(run_number) + "_" + variables['global']['model_name'] + "_optimizer_" + variables['optimizer']['type'] + "_lr_" + str(variables['optimizer']['learning_rate']).split('.')[-1] + "_epochs_" + str(variables['training']['epochs']) + "_batchsize_" + str(variables['training']['input']['batch_size']) + "_samplerTclassweightsF_" + str(variables['training']['sampler_true_class_weights_false'])  + "_scheduler_" + str(variables['training']['scheduler'])

            # setting general path
            if variables['global']['platform'] == 'colab':
                variables['global']['path'] = "exchange_base/" 
            elif variables['global']['platform'] == 'local':
                variables['global']['path'] = "/Users/mxm/Google Drive/Masterstudium/Inhalte/4th Semester/NLP/nlp-project/coding/code/exchange_base/"
            else:
                print('Please specify platform in variables')
                raise ValueError


            # Specifying directory to save output
            save_dir = variables['global']['path'] + "Model_Results/" + variables['global']['grid_search_name'] + "/" + uniqueOutputPrefix
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            variables['output']['filepath'] = save_dir

            # Path to save the dictionary of whole model performance per epoch of run
            results_json_path = variables['output']['filepath'] +   "/all_results_of_model.json"
            variables['validation']['output']['results'] = results_json_path

            # loading the vectors and labels from the exchange base in 1d or 2d
            if variables['global']['dimension_of_model'] == '1D': 
                # loading the data saved during preprocessing: 
                train_vectors = torch.load(variables['global']['path'] + "train_vectorized_1d.pt")
                train_labels = torch.load(variables['global']['path'] + "train_labels_1d.pt")

                # val set
                val_vectors = torch.load(variables['global']['path'] + "val_vectorized_1d.pt")
                val_labels = torch.load(variables['global']['path'] + "val_labels_1d.pt")

            elif variables['global']['dimension_of_model'] == '2D': 
                # loading the data saved during preprocessing: 
                train_vectors = torch.load(variables['global']['path'] + "train_vectorized_2d.pt")
                train_labels = torch.load(variables['global']['path'] + "train_labels_2d.pt")

                # val set
                val_vectors = torch.load(variables['global']['path'] + "val_vectorized_2d.pt")
                val_labels = torch.load(variables['global']['path'] + "val_labels_2d.pt")

            else:
                print('Please specify correct dimension of model.')

            # RUNNING THE MODEL
        
            # Create new object of NNSetup class
            setup = NN_Training_Setup(variables)

            # set seed everywhere for reprodrucability
            setup.setSeedEverywhere()

            # load Data into the object NNSetup
            setup.loadDataFromVariable("training",train_vectors,train_labels)
            setup.loadDataFromVariable("validation",val_vectors,val_labels)

            # Create Neural Network object based on the modules CNN_1d_experiment (if 1D) or CNN_2d_experiment (if 2D)
            if variables['global']['dimension_of_model'] == '1D': 
                model = CNN_1d_experiment(variables)

            elif variables['global']['dimension_of_model'] == '2D': 
                model = CNN_2d_experiment(variables)

            else:
                print('Please specify correct dimension of model.')
            

            # add model to NNSetup object
            setup.addNN(model)

            # define Criterion and pasting the train_labels to set class weights in case they are used
            setup.setCriterion(train_labels)

            # define Optimizer
            setup.setOptimizer()

            # set the Scheduler
            setup.setScheduler()

            # run without demo limit
            setup.train() # result can be saved automatically with dictionary and train(self,saveToFile=True)
        
            # Getting results of best epoch from this run of the grid search
            with open(results_json_path, 'r') as f:
                json_data = json.load(f)

            df = pd.DataFrame(json_data['epochs']) 
            df = df[df.stage=='validation']


            max_accuracy = df.accuracy.max()
            max_accuracy_epoch = df[df.accuracy == max_accuracy].epoch.values[0]
            max_f1_macro = df.f1_score_macro.max()
            max_f1_macro_epoch = df[df.f1_score_macro == max_f1_macro].epoch.values[0]
            max_hate_f1_score = df.f1_score_hate.max() 
            max_hate_f1_score_epoch = df[df.f1_score_hate == max_hate_f1_score].epoch.values[0]

            ResultsOverviewDict = {
                "run_number" : run_number,
                "max_accuracy" : max_accuracy,
                "max_accuracy_epoch" : max_accuracy_epoch, 
                "max_f1_macro" : max_f1_macro,
                "max_f1_macro_epoch" : max_f1_macro_epoch,
                "max_hate_f1_score" : max_hate_f1_score,
                "max_hate_f1_score_epoch" : max_hate_f1_score_epoch
            }

            # Saving results of grid search
            file = variables['global']['path'] + "Model_Results/" + variables['global']['grid_search_name'] + "/" + "ResultsOverview.csv"
            with open(file, "a+") as file:
                csv_writer = csv.DictWriter(file, fieldnames=ResultsOverviewDict.keys())
                if run_number == 0:
                    csv_writer.writeheader()
                csv_writer.writerow(ResultsOverviewDict)


