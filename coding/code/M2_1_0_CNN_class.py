import torch
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.linear import Linear
from torch.optim.adam import Adam
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot, make_dot_from_trace
import json


class CNN(nn.Module):
    # Convolutional neural network (two convolutional layers)
    def __init__(self,variables):
        """ Initializes the CNN as a class object
        """        
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential( # first layer of CNN
            nn.Conv2d(**variables["CNN"]["layers"]["1"]["Conv2d"]), 
            nn.BatchNorm2d(**variables["CNN"]["layers"]["1"]["BatchNorm2d"]),
            nn.ReLU(),
            nn.MaxPool2d(**variables["CNN"]["layers"]["1"]["MaxPool2d"])
        )
        print(self.layer1)
        self.layer2 = nn.Sequential( # second layer of CNN
            nn.Conv2d(**variables["CNN"]["layers"]["2"]["Conv2d"]), 
            nn.BatchNorm2d(**variables["CNN"]["layers"]["2"]["BatchNorm2d"]),
            nn.ReLU(),
            nn.MaxPool2d(**variables["CNN"]["layers"]["2"]["MaxPool2d"])
        )
        print(self.layer2)
        self.fc = nn.Linear(**variables["CNN"]["fc.Linear"]) # final layer of CNN
        print(self.fc)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class CNNSetup:
    def __init__(self,variables):
        self.variables = variables
        print(variables)
        self.setupGPU()

    def setupGPU(self): # Cuda config
        # If there's a GPU available...
        if torch.cuda.is_available():    
            # Tell PyTorch to use the GPU.    
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else: # If not...
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def loadFiles(self,stage):
        """ Loads tensors from the filesystem into variables which it returns
        """ 
        vectors = torch.load(self.variables[stage]["input"]["vectors"])
        labels = torch.load(self.variables[stage]["input"]["labels"])
        print("Matrix length: {:>5,}".format(len(vectors)))
        print("labels length: {:>5,}".format(len(labels)))
        return vectors, labels

    def createDataLoader(self,stage,vectors,labels):
        """ creates dataloader that allow efficient extraction
            saves these as variables in the class
        """ 
        # Combine Vectorizations with labels in TensorDataset
        dataset = TensorDataset(vectors,labels)
        # Setup PyTorch Dataloader
        dataset_loader = DataLoader(dataset,
                        #sampler = RandomSampler(dataset),
                        batch_size = self.variables[stage]["input"]["batch_size"])
        return dataset, dataset_loader

    def loadData(self,stage):
        """ wrapper for loadFiles and createDataLoader to distinguish between training
            and validation data this is done to ensure that training data is really not
            used in validation
        """ 
        vectors, labels = self.loadFiles(stage)
        dataset, dataset_loader = self.createDataLoader(stage, vectors, labels)
        if stage == "training":
            self.dataset = dataset
            self.dataset_loader = dataset_loader
        elif stage == "validation":
            self.val_dataset = dataset
            self.val_dataset_loader = dataset_loader

    def createCNN(self): # addNN(model)
        """ CNN itself is another class that has to be instanciated into a class variable
        """ 
        self.model = CNN(self.variables).to(self.device)

    def setCriterion(self):
        """ ??
        """ 
        self.criterion = nn.CrossEntropyLoss()
    
    def setOptimizer(self):
        """ ??
        """ 
        self.optimizer = torch.optim.Adam(
            params = self.model.parameters(),
            lr=self.variables["optimizer"]["learning_rate"]
        )

    def train(self,demoLimit=0):
        """ Training of the model
        """ 
        total_step = len(self.dataset_loader)
        for epoch in range(self.variables["training"]["epochs"]):
            for i, (tweetBertTensor, labels) in enumerate(self.dataset_loader):
                if (demoLimit>0) and (i>demoLimit):
                    break
                tweetBertTensor = tweetBertTensor.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(tweetBertTensor.unsqueeze(0))
                loss = self.criterion(outputs, labels)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if (i+1) % 1000 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, self.variables["training"]["epochs"], i+1, total_step, loss.item()))

    def getModel(self):
        """ returns the model
        """ 
        return self.model

    def saveModel(self):
        """ saves weights of CNN as file (really small size)
        """ 
        torch.save(self.model.state_dict(), self.variables["output"]["filepath"])
    
    def loadModel(self):
        """ loads weights saved to file
        """ 
        self.model.load_state_dict(torch.load(self.variables["validation"]["input"]["model"]))


    def evaluate(self):
        """ uses another dataset to calculate accuracy of model
        """ 
        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for tweetBertTensor, labels in self.val_dataset_loader:
                
                tweetBertTensor = tweetBertTensor.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(tweetBertTensor.unsqueeze(0))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test tweetBertTensor: {} %'.format(100 * correct / total))

            # TODO F1 score pro class
            # TODO F1 macro score (average for all classes)
            result = {
                "correct" : correct,
                "total" : total,
                "accuracy" : 100*correct/total
            }

            return result
    
    def saveEvaluation(self,result,id):
        filenpath = self.variables["validation"]["input"]["result"] + id + ".json"
        with open(filenpath, 'w') as fp:
                json.dump(result, fp)


if __name__ == "__main__":
    # prefix to test different setups
    uniqueInputPrefix = ""
    uniqueOutputPrefix = "test1_"
    path = "coding/code/exchange_base/"
    # Training input
    stage = "train"
    train_filpath_vectors = path + uniqueInputPrefix + stage +  "_vectorized.pt"
    train_filepath_labels = path + uniqueInputPrefix + stage +  "_labels.pt"
    # Model Training
    epochs = 3
    # Model Output
    output_filepath_model = path + uniqueOutputPrefix + stage + "_model_epochs" + str(epochs) + ".ckpt"
    # Evaluation
    stage = "val"
    val_filepath_vectors = path + uniqueInputPrefix + stage +  "_vectorized.pt"
    val_filepath_labels = path + uniqueInputPrefix + stage +  "_labels.pt"
    val_filepath_result_prefix = path + uniqueOutputPrefix + stage +  "_result_"

    variables =	{
        "global" : {
            "path" : path
        },
        "CNN" : {
            "layers" : {
                "1" : {
                    "Conv2d" : {
                        "in_channels" : 1,
                        "out_channels" : 16,
                        "kernel_size" : 3,
                        "stride" : 1,
                        "padding" : 2,
                    },
                    "BatchNorm2d" : {
                        "num_features" : 16
                    },
                    "MaxPool2d" : {
                        "kernel_size" : 2,
                        "stride" : 2
                    }
                },
                "2" : {
                    "Conv2d" : {
                        "in_channels" : 16,
                        "out_channels" : 32,
                        "kernel_size" : 5,
                        "stride" : 1,
                        "padding" : 2
                    },
                    "BatchNorm2d" : {
                        "num_features" : 32
                    },
                    "MaxPool2d" : {
                        "kernel_size" : 2,
                        "stride" : 2
                    }
                }
            },
            "fc.Linear" : {
                "in_features" : 288,
                "out_features" : 3
            }
        },
        "optimizer" : {
            "learning_rate" : 0.001
        },
        "training" : {
            "epochs" : epochs,
            "input" : {
                "batch_size": 1,
                "vectors": train_filpath_vectors,
                "labels": train_filepath_labels
            },
        },
        "output" : {
            "filepath" : output_filepath_model
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

    # setup = CNNSetup(variables)
    # setup.loadData("training")
    # setup.createCNN()
    # setup.setCriterion()
    # setup.setOptimizer()
    # setup.train(demoLimit=1000)
    # setup.saveModel()
    # # setup.loadModel() # only necessary when just evaluation models
    # setup.loadData("validation")
    # setup.saveEvaluation(setup.evaluate(),"final")


    class betterCNNSetup(CNNSetup):
        def __init__(self,variables):
            CNNSetup.__init__(self,variables)
        
        def train(self,demoLimit=0):
            """ Training of the model
            """ 
            total_step = len(self.dataset_loader)
            for epoch in range(self.variables["training"]["epochs"]):
                for i, (tweetBertTensor, labels) in enumerate(self.dataset_loader):
                    if (demoLimit>0) and (i>demoLimit):
                        break
                    tweetBertTensor = tweetBertTensor.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(tweetBertTensor.unsqueeze(0))
                    loss = self.criterion(outputs, labels)
                    
                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    if (i+1) % 1000 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch+1, self.variables["training"]["epochs"], i+1, total_step, loss.item()))
                self.saveEvaluation(self.evaluate(),"epoch"+str(epoch))
        
        def evaluate(self):
            """ uses another dataset to calculate accuracy of model
            """ 
            # gets executed after each epoch
            self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
            with torch.no_grad():
                # full validation data set
                correct = 0
                total = 0
                for tweetBertTensor, labels in self.val_dataset_loader:
                    # Batch with one tweet
                    tweetBertTensor = tweetBertTensor.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(tweetBertTensor.unsqueeze(0))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    

                print('Test Accuracy of the model on the 10000 test tweetBertTensor: {} %'.format(100 * correct / total))

                # TODO F1 score pro class
                # TODO F1 macro score (average for all classes)
                print("F1 score here")
                result = {
                    "correct" : correct,
                    "total" : total,
                    "accuracy" : 100*correct/total
                }

                return result

    #variables_current = variables
    #variables_current["optimizer"]["learning_rate"] = 0.002

    setup2 = betterCNNSetup(variables)
    setup2.loadData("training")
    setup2.loadData("validation")
    setup2.createCNN()
    setup2.setCriterion()
    setup2.setOptimizer()
    setup2.train(demoLimit=3000)
    setup2.saveModel()
    # setup.loadModel() # only necessary when just evaluation models
