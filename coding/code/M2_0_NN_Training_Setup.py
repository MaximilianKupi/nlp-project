"""This is the script to define the class for setting up the training and evaluation routine.
"""

# loading the required packages
import torch
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import ELU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.linear import Linear
from torch.optim.adam import Adam
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot, make_dot_from_trace
import json
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import os


class NN_Training_Setup:
    """Sets up the main training and validation routine for the model.
    
    Attributes:
        variables: The dictionary containing the variables to specify the training routine (see MAIN module).
    """

    def __init__(self,variables):
        self.variables = variables
        #print(variables)
        self.setupGPU()

    def setupGPU(self):
        """Checks if GPU is available and tells PyTorch to use the GPU if available.
        
        Returns: 
            Prints status report.
        """
        # Cuda config
        # If there's a GPU available...
        if torch.cuda.is_available():    
            # Tell PyTorch to use the GPU.    
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else: # If not...
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def setSeedEverywhere(self):
        """Sets seed for all random processes.
        """
        np.random.seed(self.variables['global']["seed"])
        torch.manual_seed(self.variables['global']["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.variables['global']["seed"])


    def loadFiles(self,stage):
        """Loads tensors from the filesystem into variables which it returns.

        Args:
            stage (str): The state which the model is in with respect to the specific file names (here 'validation', 'training', 'testing')
        
        Returns:
            The variables for vectors and labels as torch tensors.
        """ 
        vectors = torch.load(self.variables[stage]["input"]["vectors"])
        labels = torch.load(self.variables[stage]["input"]["labels"])
        print("Matrix length: {:>5,}".format(len(vectors)))
        print("labels length: {:>5,}".format(len(labels)))
        return vectors, labels
  

    def createDataLoader(self,stage,vectors,labels, shuffle=True, sampler=True):
        """Creates dataloader that allows efficient extraction of samples.
            
        Args:
            stage (str): The state which the model is in with respect to the specific file names (here 'validation', 'training', 'testing')
            vectors (torch tensor): The representation of tweet as tensor (is matrix in 2D case).
            labels (torch tensor): The labels of each tweet as long tensor.
            shuffle (bool): Whether or not to use the shuffler while training the model. Default: True.
            sampler (bool): Whether or not to use the sampler while training the model. The sampler takes into account the sample weights such that each class is represented equally while training. Default: True.
        
        Returns:
            Dataset and data loader. 
        """ 
        # Combine Vectorizations with labels in TensorDataset
        dataset = TensorDataset(vectors,labels)
        #print('dataset size:')
        #print(dataset)

        # compute the weights
        targets = []
        for _, target in dataset:
                targets.append(target)
        targets = torch.stack(targets)

        # Compute samples weight (each sample should get its own weight)
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in targets])

        # Create sampler
        if sampler:
            sampler_object = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        else:
            sampler_object = None

        # Setup PyTorch Dataloader
        dataset_loader = DataLoader(dataset,
                        shuffle=shuffle,
                        sampler = sampler_object,
                        batch_size=self.variables[stage]["input"]["batch_size"])
        return dataset, dataset_loader

    def saveDataToVariables(self,stage,vectors,labels):
        """Saves the dataset and dataloader for training (self.dataset, self.dataset_loader) and validation (self.val_dataset, self.val_dataset_loader) into class variables.

        Args:
            stage (str): The state which the model is in with respect to the specific file names (here 'validation', 'training', 'testing')
            vectors (torch tensor): The representation of tweet as tensor (is matrix in 2D case).
            labels (torch tensor): The labels of each tweet as long tensor.
        """
        vectors = vectors.float()
        labels = labels.type(torch.LongTensor)
        #labels = labels.float()
        print("Demo Vector entry")
        print(vectors[0])
        print("Demo Label entry")
        print(labels[0])
        if stage == "training":
            if self.variables['training']['sampler_true_class_weights_false']:
                self.dataset, self.dataset_loader = self.createDataLoader(stage, vectors, labels, shuffle=False, sampler=True)
            else:
                self.dataset, self.dataset_loader = self.createDataLoader(stage, vectors, labels, shuffle=True, sampler=False)
        
        elif stage == "validation":
            self.val_dataset, self.val_dataset_loader = self.createDataLoader(stage, vectors, labels, shuffle=False, sampler=False)
        
        elif stage == "testing":
            self.test_dataset, self.test_dataset_loader = self.createDataLoader(stage, vectors, labels, shuffle=False, sampler=False)

    def loadDataFromVariable(self,stage,vectors,labels):
        """Wrapper for createDataLoader.

        Uses input data and distinguishes between training and validation data (see MAIN file). 
        This is done to ensure that training data is really not used in validation.
        Saves object as class variable. 

        Args:
            stage (str): The state which the model is in with respect to the specific file names (here 'validation', 'training', 'testing').
            vectors (torch tensor): The representation of tweet as tensor (is matrix in 2D case).
            labels (torch tensor): The labels of each tweet as long tensor.
        """ 
        self.saveDataToVariables(stage,vectors,labels)


    def addNN(self,model):
        """Adds any model as class variable to the NN_training_Setup object (and sends it to the GPU if available), so that it can be used in the training / validation routine. 
        """
        self.model = model.to(self.device)

    def setCriterion(self, labels):
        """Sets the loss function to cross entropy loss (since we have a multi class problem). 
        
        Listens to variable "sampler_true_class_weights_false" from the dictionary to only use class weights if sampler is set to false.
        Saves the loss function for training and validation in a class variable. 

        Args:
            labels (torch tensor): The labels of each tweet as long tensor to calculate the class weights.
        """ 
        # we set weights for training if so specified in variables
        if not self.variables['training']['sampler_true_class_weights_false']:
            # calculating the weights per class
            labels = labels.numpy()
            print(labels)
            label_unique = np.unique(labels)
            print(label_unique)
            class_weights = class_weight.compute_class_weight('balanced', label_unique, labels)
            print('Class Weights:', class_weights)
            class_weights = torch.from_numpy(class_weights).cuda().float()
            # setting the criterion for the loss function
            self.train_criterion = nn.CrossEntropyLoss(weight=class_weights)
        else: 
            self.train_criterion = nn.CrossEntropyLoss()
        # for validation we don't use weights
        self.val_criterion = nn.CrossEntropyLoss()
    
    def setOptimizer(self):
        """ Sets the optimizer to Adam, RMSprop, or SGD, depending on the specification in the variables from the dictionary.
        
        Applies the respective learning rates from the dictionary, as well as the momentum in case of an SGD optimizer.
        Saves the optimizer into a class variable.
        
        Warns if no valid optimizer is specified.
        """ 

        if self.variables['optimizer']['type'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                params = self.model.parameters(),
                lr=self.variables["optimizer"]["learning_rate"],
                amsgrad=True,
            )

        elif self.variables['optimizer']['type'] == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(
                params = self.model.parameters(),
                lr=self.variables["optimizer"]["learning_rate"]
            )

        elif self.variables['optimizer']['type'] == 'SGD':
                self.optimizer = torch.optim.SGD(
                    params = self.model.parameters(),
                    lr=self.variables["optimizer"]["learning_rate"],
                    momentum= self.variables['optimizer']['momentum']
                )
        else:
            print('Please specify a valid optimizer (Adam, RMSprop, or SGD).')
    
    def setScheduler(self, mode='max', factor=0.1, patience=2):
        """Sets the scheduler so that the learning rate is reduced dynamically based on the validation measures.

        Args:
            mode (str): One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘max’.
            factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
            patience (int): Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 2.
        """
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode=mode, factor=factor, patience=patience, verbose=True)


    def getAccuracy(self, total, correct):
        """Calculates the accuracy based on the number of correctly predicted classes.

        Args:
            total (int): Total number of tweets to predict.
            correct (int): Number of correctly predicted tweets.
        
        Returns:
            Accuracy based on total tweets and number of correctly predicted tweets.
        """
        return 100*(correct/total)

    def train(self,demoLimit=0,saveToFile=True):
        """Trains the model.

        Saves the model as well as the training / validation metrics into a class variable. 

        Args:
            demoLimit (int): Sets a demo limit to reduce the dataset for demonstration / testing purposes only. Default: 0.
            saveToFile (bool): Whether or not to save the model as well as the training / validation metrics to a file. Default: True.
        
        Returns: 
            The model as well as the training / validation metrics as dictionary. 

        Warning:
            Don't use demoLimit during actual training routine – only meant for test purposes. 
        """ 

        self.resetResultMemory()

        total_step = len(self.dataset_loader)

        for epoch in range(self.variables["training"]["epochs"]):
            correct = 0
            total = 0
            outputs_epoch = []
            labels_epoch = []
            predicted_epoch = []
            running_acc = 0.
            running_loss = 0.

            for i, (labelsBertTensor, labels) in enumerate(self.dataset_loader):
                if (demoLimit>0) and (i>demoLimit):
                    break
                labelsBertTensor = labelsBertTensor.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(self.prepareVectorForNN(labelsBertTensor))

                _, predicted = torch.max(outputs.data, 1)

                # calculating total number, correct number and loss for predicted tweets
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = self.train_criterion(outputs, labels)

                # appending the overall predicted and target tensor for the whole epoch to calculate the metrics as lists
                labels_epoch.append(labels)
                predicted_epoch.append(predicted)

                # calculating the running loss
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (i+1)

                # calculating the running accuracy
                acc_t = self.getAccuracy(total, correct)
                running_acc += (acc_t - running_acc) / (i+1)


                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if (i+1) % 1000 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, self.variables["training"]["epochs"], i+1, total_step, loss.item()))
            
            # reconverting lists to tensors
            labels_epoch = torch.cat(labels_epoch).cpu()
            predicted_epoch = torch.cat(predicted_epoch).cpu()     

            # calculating the classification report with sklearn    
            classification_report_json = classification_report(labels_epoch, predicted_epoch, output_dict=True)
            classification_report_str = classification_report(labels_epoch, predicted_epoch, output_dict=False)
            
            # deleting comas inbetween numbers to write it into the json file
            labels_epoch_str =  " ".join(str(x) for x in labels_epoch.numpy().tolist()) 
            predicted_epoch_str = " ".join(str(x) for x in predicted_epoch.numpy().tolist())

            result = {
                "epoch" : epoch,
                "correct" : correct,
                "total" : total,
                "accuracy" : running_acc, 
                "loss" : running_loss, 
                "stage" : 'training',
                "f1_score_hate" : classification_report_json['0']['f1-score'],
                "f1_score_macro" : classification_report_json['macro avg']['f1-score'],
                "classification_report_json" : classification_report_json,
                "classification_report_str" : classification_report_str,
                "predicted_epoch": predicted_epoch_str,
                "labels_epoch": labels_epoch_str
                }

            # saving results of evaluation on training set    
            self.saveEvaluation(result, epoch)

            # evaluating on validation set and saving results
            self.saveEvaluation(self.evaluate(epoch), epoch)

            # saving the model after each epoch
            save_dir = self.variables["output"]["filepath"]
            save_prefix = os.path.join(save_dir, 'Model' )
            save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
            print("save model to {}".format(save_path))
            with open(save_path, mode="wb") as output:
                torch.save(self.model.state_dict(), output)


            # setting the scheduler to dynamically adapt the learning rate based on the f1-score macro
            if self.variables['training']['scheduler']:
                self.scheduler.step(classification_report_json['macro avg']['f1-score'])


        if saveToFile:
            self.writeResultMemoryToFile()
        else:
            return self.getResult()

    def prepareVectorForNN(self,vector):
        """Prepares the input tensor for CNN by unsqueezing it on the first dimension. This is where the features will be written to while convoluting.

        Args:
            vector (torch tensor): Input tensor for the model.

        Returns:
            Unsqueezed input tensor. 
        """
        #print(str(vector.size()))
        vector1 =  vector.unsqueeze(1)
        #print(str(vector1.size()))
        return vector1

    def evaluate(self, epoch):
        """Evaluates the model on the validation dataset.

        Args:
            epoch (int): The current epoch number (is used to write the results).

        Returns:
            The evaluation results.
        """ 

        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            outputs_epoch = []
            labels_epoch = []
            predicted_epoch = []
            running_acc = 0.
            running_loss = 0.

            for i, (labelsBertTensor, labels) in enumerate(self.val_dataset_loader):
                
                labelsBertTensor = labelsBertTensor.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(self.prepareVectorForNN(labelsBertTensor))
                #print(outputs.data)
                _, predicted = torch.max(outputs.data, 1)
                #print(predicted.item())

                # calculating total number, correct number and loss for predicted tweets
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = self.val_criterion(outputs, labels)

                # appending the overall predicted and target tensor for the whole epoch to calculate the metrics as lists
                labels_epoch.append(labels)
                predicted_epoch.append(predicted)

                # calculating the running loss
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (i+1)

                # calculating the running accuracy
                acc_t = self.getAccuracy(total, correct)
                running_acc += (acc_t - running_acc) / (i+1)

            # tretransforming the list into tensors
            labels_epoch = torch.cat(labels_epoch).cpu()
            predicted_epoch = torch.cat(predicted_epoch).cpu()
            
            # calculating the classification report with sklearn    
            classification_report_json = classification_report(labels_epoch, predicted_epoch, output_dict=True)
            classification_report_str = classification_report(labels_epoch, predicted_epoch, output_dict=False)
            
            labels_epoch_str =  " ".join(str(x) for x in labels_epoch.numpy().tolist()) 
            predicted_epoch_str = " ".join(str(x) for x in predicted_epoch.numpy().tolist())

            result = {
                "epoch" : epoch,
                "correct" : correct,
                "total" : total,
                "accuracy" : running_acc, 
                "loss" : running_loss, 
                "stage" : 'validation',
                "f1_score_hate" : classification_report_json['0']['f1-score'],
                "f1_score_macro" : classification_report_json['macro avg']['f1-score'],
                "classification_report_json" : classification_report_json,
                "classification_report_str" : classification_report_str,
                "predicted_epoch": predicted_epoch_str,
                "labels_epoch": labels_epoch_str
                }

            print('Test Accuracy on Validation: {} %'.format(running_acc))
            print('F1-score Macro on Validation: {}'.format(classification_report_json['macro avg']['f1-score']))
            print('Precision of Hate on Validation: {}'.format(classification_report_json['0']['precision']))
            print('Recall of Hate on Validation: {}'.format(classification_report_json['0']['recall']))
            print('F1-score of Hate on Validation: {}'.format(classification_report_json['0']['f1-score']))

            return result
    
    def resetResultMemory(self):
        """Resets the memory where the results get written into.
        """
        self.result = {
                "epochs" : [],
                "variables" : self.variables
            }

    def saveEvaluation(self,result,epoch):
        """Saves the evaluation to the results.
        """
        self.result['epochs'].append(result)
    
    def getResult(self):
        """Gets the results.

        Returns:
            The results as class variable.
        """
        return self.result

    def writeResultMemoryToFile(self):
        """Writes results to file.
        
        Takes the file path and name based on the combination of training specifications in the variables. 
        """
        file = self.variables['validation']['output']['results'] 
        with open(file, 'w+', encoding='utf-8') as fp:
            print("Save outputfile")
            print(file)
            json.dump(self.result, fp, separators=(',', ':'), indent=4)

    def loadModel(self, model_path):
            """Loads pretrained model for testing.

            Args:
                model_path: The location of the model to test.
            """ 
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def testModel(self):
            """Evaluates the model on the validation dataset.

            Returns:
                The evaluation results.
            """ 

            self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
            with torch.no_grad():
                correct = 0
                total = 0
                outputs_epoch = []
                labels_epoch = []
                predicted_epoch = []
                running_acc = 0.
         
                for i, (labelsBertTensor, labels) in enumerate(self.test_dataset_loader):
                    
                    labelsBertTensor = labelsBertTensor.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(self.prepareVectorForNN(labelsBertTensor))
                    #print(outputs.data)
                    _, predicted = torch.max(outputs.data, 1)
                    #print(predicted.item())

                    # calculating total number, correct number and loss for predicted tweets
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # appending the overall predicted and target tensor for the whole epoch to calculate the metrics as lists
                    labels_epoch.append(labels)
                    predicted_epoch.append(predicted)

                    # calculating the running accuracy
                    acc_t = self.getAccuracy(total, correct)
                    running_acc += (acc_t - running_acc) / (i+1)

                # tretransforming the list into tensors
                labels_epoch = torch.cat(labels_epoch).cpu()
                predicted_epoch = torch.cat(predicted_epoch).cpu()
                
                # calculating the classification report with sklearn    
                classification_report_json = classification_report(labels_epoch, predicted_epoch, output_dict=True)
                classification_report_str = classification_report(labels_epoch, predicted_epoch, output_dict=False)
                
                labels_epoch_str =  " ".join(str(x) for x in labels_epoch.numpy().tolist()) 
                predicted_epoch_str = " ".join(str(x) for x in predicted_epoch.numpy().tolist())

                result = {
                    "epoch" : 0,
                    "correct" : correct,
                    "total" : total,
                    "accuracy" : running_acc, 
                    "stage" : 'testing',
                    "f1_score_hate" : classification_report_json['0']['f1-score'],
                    "f1_score_macro" : classification_report_json['macro avg']['f1-score'],
                    "classification_report_json" : classification_report_json,
                    "classification_report_str" : classification_report_str,
                    "predicted_epoch": predicted_epoch_str,
                    "labels_epoch": labels_epoch_str
                    }

                print('Test Accuracy on Testing: {} %'.format(running_acc))
                print('F1-score Macro on Testing: {}'.format(classification_report_json['macro avg']['f1-score']))
                print('Precision of Hate on Testing: {}'.format(classification_report_json['0']['precision']))
                print('Recall of Hate on Testing: {}'.format(classification_report_json['0']['recall']))
                print('F1-score of Hate on Testing: {}'.format(classification_report_json['0']['f1-score']))

                # Saving the testing results to a file

                file_path = self.variables['testing']['output']['results'] 
                with open(file_path, 'w+', encoding='utf-8') as fp:
                    print("Save outputfile")
                    print(file_path)
                    json.dump(result, fp, separators=(',', ':'), indent=4)