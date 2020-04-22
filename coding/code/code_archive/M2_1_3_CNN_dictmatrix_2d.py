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


######################
## Data loading     ##
######################
# path = "coding/code/exchange_base/"
# stage = "train"
# input_file_name_vectorized = path + stage +  "_vectorized.pt"
# input_file_name_labels = path + stage +  "_labels.pt"


def setupGPU():
    ######################
    ## CUDA config      ##
    ######################
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

def loadData(input_file_name_vectorized, input_file_name_labels):
    vectors = torch.load(input_file_name_vectorized)
    # Loading the label tensor
    labels = torch.load(input_file_name_labels)
    print("Matrix length: {:>5,}".format(len(vectors)))
    print("labels length: {:>5,}".format(len(labels)))
    return vectors, labels

def createDataLoader(vectors, labels, batch_size):
    # Combine Vectorizations with labels in TensorDataset
    dataset = TensorDataset(vectors,labels)
    # Setup PyTorch Dataloader
    dataset_loader = DataLoader(dataset,
                    #sampler = RandomSampler(dataset),
                    batch_size = batch_size)
    return dataset_loader

def dataloaderFromFiles(input_file_name_vectorized,input_file_name_labels,batch_size):
    vectors, labels = loadData(input_file_name_vectorized, input_file_name_labels)
    dataset_loader = createDataLoader(vectors,labels,batch_size)
    return dataset_loader

# Convolutional neural network (two convolutional layers)
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), #kernel_size=3, stride=1, padding=2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), #kernel_size=5, stride=1, padding=2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(192, num_classes) #7*7*32
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out



# Output model graphviz
#graph = make_dot(model(tweetBertTensor.unsqueeze(0)), params=dict(model.named_parameters()))

def Training(): #copy this to jupyter for controlled execution
    device = setupGPU()
    ######################
    ## Configuration    ##
    ######################
    # Batch Size for DataLoader
    batch_size = 1
    num_epochs = 5
    num_classes = 3
    learning_rate = 0.001

    path = "coding/code/exchange_base/"
    stage = "train"
    input_file_name_vectorized = path + stage +  "_vectorized.pt"
    input_file_name_labels = path + stage +  "_labels.pt"

    dataset_loader = dataloaderFromFiles(input_file_name_vectorized, input_file_name_labels, batch_size)

    model = CNN(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(dataset_loader)
    for epoch in range(num_epochs):
        for i, (tweetBertTensor, labels) in enumerate(dataset_loader):
            tweetBertTensor = tweetBertTensor.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(tweetBertTensor.unsqueeze(0))
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 1000 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    return model


def evaluation(model): #copy this to jupyter notebook after training for controlled evaluation
    device = setupGPU()
    path = "coding/code/exchange_base/"
    stage = "val"
    input_file_name_vectorized_val = path + stage +  "_vectorized.pt"
    input_file_name_labels_val = path + stage +  "_labels.pt"

    dataset_loader = dataloaderFromFiles(input_file_name_vectorized_val, input_file_name_labels_val,batch_size)

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for tweetBertTensor, labels in dataset_loader:
            
            tweetBertTensor = tweetBertTensor.to(device)
            labels = labels.to(device)
            outputs = model(tweetBertTensor.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test tweetBertTensor: {} %'.format(100 * correct / total))


if __name__ == "__main__":
    # Running the function
    Training()
    
    # Save model checkpoint
    torch.save(model.state_dict(), path +"model" + "_epochs" + str(num_epochs) + ".ckpt")