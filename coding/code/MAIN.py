#### IMPORTING PACKAGES ####

## packages used in the MAIN file
import pandas as pd
import json
import platform
from sklearn.model_selection import ParameterGrid
import csv 

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

# # class to setup the model
#from M2_1_CNN_1d import CNN_1d
# since we are still in the experimenting mode we use the CNN experiment
from M2_1_CNN_1d_experiment import CNN_1d_experiment
from M2_1_CNN_2d_experiment import CNN_2d_experiment


# # class to setup the dataloading, the training and the evaluation  
from M2_0_NN_Training_Setup import *



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
        "platform": 'local', # 'local' 'colab'
        "model_name" : "CNN_experiment",
        "grid_search_name" : "Retraining_Best_Performing_Model_90_epochs",
        "dimension_of_model" : "1D" #2D
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
    "output": {
        "results" : {
        #will be filled on the go
        },
    },
    # currently not used since we are using cnn experiment
    # "CNN" : {
    #     "layers" : {
    #         "1" : {
    #             "Conv1d" : {
    #                 "in_channels" : 120,
    #                 "out_channels" : 16,
    #                 "kernel_size" : 3
    #             },
    #             "BatchNorm1d" : {
    #                 "num_features" : 16
    #             },
    #             "MaxPool2d" : {
    #                 "kernel_size" : 2,
    #                 "stride" : 2
    #             }
    #         },
    #         "2" : {
    #             "Conv1d" : {
    #                 "in_channels" : 16,
    #                 "out_channels" : 32,
    #                 "kernel_size" : 3,
    #             },
    #             "BatchNorm1d" : {
    #                 "num_features" : 32
    #             },
    #             "MaxPool2d" : {
    #                 "kernel_size" : 2,
    #                 "stride" : 2
    #             }
    #         }
    #     },
    #     "fc.Linear" : {
    #         "in_features" : 32,
    #         "out_features" : 3
    #     }
    # },
}

# SETTING-UP HYPER PARAMETER GRID SEARCH
param_grid = {  'optimizer_type':variables['optimizer']['type'], 
                'learning_rate': variables['optimizer']['learning_rate'], 
                "sampler_true_class_weights_false" : variables['training']["sampler_true_class_weights_false"],
                "scheduler" : variables['training']["scheduler"]}

all_params = list(ParameterGrid(param_grid))
 
# Running through the grid
#TODO: Add run Run Number 27 to Results Overview.scv

# protecting things from being run during documentation process:
if __name__ == "__main__":
    
    for run_number, current_params in enumerate(all_params):
        if run_number != 26:
            print('skipping', run_number)
        else:

            print(current_params)
            # writing the parameters of that grid search run into the dictionary
            variables['optimizer']['learning_rate'] = current_params['learning_rate']
            variables['optimizer']['type'] = current_params['optimizer_type']
            variables['training']["sampler_true_class_weights_false"] = current_params["sampler_true_class_weights_false" ]
            variables['training']["scheduler"] = current_params['scheduler']


            # SPECIFYING  PREFIXES AND FILEPATHS 

            uniqueInputPrefix = ""
            uniqueOutputPrefix = str(run_number) + "_" + variables['global']['model_name'] + "_optimizer_" + variables['optimizer']['type'] + "_lr_" + str(variables['optimizer']['learning_rate']).split('.')[-1] + "_epochs_" + str(variables['training']['epochs']) + "_batchsize_" + str(variables['training']['input']['batch_size']) + "_samplerTclassweightsF_" + str(variables['training']['sampler_true_class_weights_false'])  + "_scheduler_" + str(variables['training']['scheduler'])

            # setting general path
            if variables['global']['platform'] == 'colab':
                variables['global']['path'] = "exchange_base/" 
            elif variables['global']['platform'] == 'local':
                variables['global']['path'] = "/Users/mxm/Google Drive/Masterstudium/Inhalte/4th Semester/NLP/nlp-project/coding/code/exchange_base/"
            else:
                print('Please specify platform in variables')
                raise ValueError


            # Training Output
            save_dir = variables['global']['path'] + "Model_Results/" + variables['global']['grid_search_name'] + "/" + uniqueOutputPrefix
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            variables['output']['filepath'] = save_dir

            # Evaluation path to save the dictionary of whole model performance in run
            results_json_path = variables['output']['filepath'] +   "/all_results_of_model.json"
            variables['validation']['output']['results'] = results_json_path

            # loading the vectors and labels from the exchange base
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

            # load Data into the object NNSetup
            setup.loadDataFromVariable("training",train_vectors,train_labels)
            setup.loadDataFromVariable("validation",val_vectors,val_labels)

            # Create Neural Network object from class nn.module
            if variables['global']['dimension_of_model'] == '1D': 
                model = CNN_1d_experiment(variables)

            elif variables['global']['dimension_of_model'] == '2D': 
                model = CNN_2d_experiment(variables)

            else:
                print('Please specify correct dimension of model.')
            


            # Create Neural Network object with the model from class
            #model = CNN_1d_experiment(initial_num_channels=1, num_channels=256, hidden_dim=256, num_classes=3, dropout_p=0.1)

            # add model to NNSetup object
            setup.addNN(model)

            # define Criterion
            setup.setCriterion(train_labels)

            # define Optimizer
            setup.setOptimizer()

            # set the Scheduler
            setup.setScheduler()

            # run with demo limit
            #result = setup.train(demoLimit=5000, saveToFile=True) # result can be saved automatically with dictionary and train(self,saveToFile=True)

            # run without demo limit
            setup.train() # result can be saved automatically with dictionary and train(self,saveToFile=True)
        
            # Getting results of best epoch from this run
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


