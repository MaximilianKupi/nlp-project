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
from M2_1_CNN import CNN
import os


class NNSetup:
    """ The main working horse to setup our the training routine """

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

    # def make_weights_for_balanced_classes(labels, nclasses):
    #     """ This method generates the weights to balance the dataset while training
    #     """                        
    #     count = [0] * nclasses                                                      
    #     for item in enumerate(labels):                                                         
    #         count[item[1]] += 1                                                     
    #     weight_per_class = [0.] * nclasses                                      
    #     N = float(sum(count))                                                   
    #     for i in range(nclasses):                                                   
    #         weight_per_class[i] = N/float(count[i])                                 
    #     weight = [0] * len(labels)                                              
    #     for idx, val in enumerate(labels):                                          
    #         weight[idx] = weight_per_class[val[1]]                                  
    #     return weight  

    def createDataLoader(self,stage,vectors,labels, shuffle=True, sampler=True):
        """ creates dataloader that allow efficient extraction
            saves these as variables in the class
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
        vectors = vectors.float()
        labels = labels.type(torch.LongTensor)
        #labels = labels.float()
        print("Demo Vector entry")
        print(vectors[0])
        print("Demo Label entry")
        print(labels[0])
        if stage == "training":
            if self.variables['training']['sampler']:
                self.dataset, self.dataset_loader = self.createDataLoader(stage, vectors, labels, shuffle=False, sampler=True)
            else:
                self.dataset, self.dataset_loader = self.createDataLoader(stage, vectors, labels, shuffle=True, sampler=False)
        
        elif stage == "validation":
            self.val_dataset, self.val_dataset_loader = self.createDataLoader(stage, vectors, labels, shuffle=False, sampler=False)


    def loadDataFromVariable(self,stage,vectors,labels):
        """ wrapper for createDataLoader, uses input data and distinguishes between training
            and validation data this is done to ensure that training data is really not
            used in validation
        """ 
        self.saveDataToVariables(stage,vectors,labels)

    def loadData(self,stage):
        """ wrapper for loadFiles and createDataLoader to distinguish between training
            and validation data this is done to ensure that training data is really not
            used in validation
        """ 
        vectors, labels = self.loadFiles(stage)
        self.saveDataToVariables(stage,vectors,labels)
        

    def createCNN(self): # addNN(model)
        """ CNN itself is another class that has to be instanciated into a class variable
        """ 
        self.model = CNN(self.variables).to(self.device)

    def addNN(self,model):
        self.model = model.to(self.device)

    def setCriterion(self):
        """ Setting the loss function to cross entropy loss since we have a multi class problem
        """ 
        self.criterion = nn.CrossEntropyLoss()
    
    def setOptimizer(self):
        """ Setting the optimizer to Adam as this is the state of the art optimizer for these kind of tasks.
        """ 

        if self.variables['training']['optimizer']['type'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                params = self.model.parameters(),
                lr=self.variables["optimizer"]["learning_rate"],
                amsgrad=True,
            )

        elif self.variables['training']['optimizer']['type'] == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(
                params = self.model.parameters(),
                lr=self.variables["optimizer"]["learning_rate"]
            )

        elif self.variables['training']['optimizer']['type'] == 'SGD':
                self.optimizer = torch.optim.SGD(
                    params = self.model.parameters(),
                    lr=self.variables["optimizer"]["learning_rate"],
                    moomentum= self.variables['training']['optimizer']['momentum']
                )
        else:
            print('Please specify a valid optimizer')
    
    def setScheduler(self):
        """ Setting the scheduler so that the learning rate is reduced dynamically based on the validation measures.
        """
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.5, patience=1)


    def getAccuracy(self, total, correct):
        """ Calculating the accuracy based on the number of correctly predicted classes.
        """
        return 100*(correct/total)

    def train(self,demoLimit=0,saveToFile=False):
        """ Training of the model
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

                if self.variables['training']['softmax']:
                    _, predicted = torch.softmax(outputs.data, 1)
                else: 
                    _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = self.criterion(outputs, labels)

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
                
            result = {
                "epoch" : epoch,
                "correct" : correct,
                "total" : total,
                "accuracy" : running_acc, 
                "loss" : running_loss, 
                "stage" : 'training',
                "recall_hate" : classification_report_json['0']['recall'],
                "f1-score macro" : classification_report_json['macro avg']['f1-score'],
                "classification_report_json" : classification_report_json,
                "classification_report_str" : classification_report_str,
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
            # torch.save(model.state_dict(), save_path)
            # setting the scheduler to dynamically adapt the learning rate based on the f1-score macro
            if self.variables['training']['scheduler']:
                self.scheduler.step(classification_report_json['macro avg']['f1-score'])
            #TODO: Add option to turn the scheduler on if needed

        #TODO: Save Best Model metric: classification_report_json['macro avg']['f1-score']
        #TODO: Implement early stopping rule

        if saveToFile:
            self.writeResultMemoryToFile()
        else:
            return self.getResult()

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

    def prepareVectorForNN(self,vector):
        #print(str(vector.size()))
        vector1 =  vector.unsqueeze(1)
        #print(str(vector1.size()))
        return vector1

    def evaluate(self, epoch):
        """ uses another dataset to calculate accuracy of model
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
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = self.criterion(outputs, labels)

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
                
            result = {
                "epoch" : epoch,
                "correct" : correct,
                "total" : total,
                "accuracy" : running_acc, 
                "loss" : running_loss, 
                "stage" : 'validation',
                "recall_hate" : classification_report_json['0']['recall'],
                "f1-score macro" : classification_report_json['macro avg']['f1-score'],
                "classification_report_json" : classification_report_json,
                "classification_report_str" : classification_report_str,
                }

            print('Test Accuracy on Validation: {} %'.format(running_acc))
            print('F1-score Macro on Validation: {}'.format(classification_report_json['macro avg']['f1-score']))
            print('Precision of Hate on Validation: {}'.format(classification_report_json['0']['precision']))
            print('Recall of Hate on Validation: {}'.format(classification_report_json['0']['recall']))
            print('F1-score of Hate on Validation: {}'.format(classification_report_json['0']['f1-score']))

            return result
    
    def resetResultMemory(self):
        self.result = {
                "epochs" : [],
                "variables" : self.variables
            }

    def saveEvaluation(self,result,epoch):
        self.result['epochs'].append(result)
    
    def getResult(self):
        return self.result

    def writeResultMemoryToFile(self):
        filenpath = self.variables["validation"]["input"]["result"]
        with open(filenpath, 'w+', encoding='utf-8') as fp:
            print("Save outputfile")
            json.dump(self.result, fp, separators=(',', ':'), indent=4)



