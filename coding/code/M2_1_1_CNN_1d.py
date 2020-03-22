# %%
import torch
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import RandomSampler

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
from torch.utils.data.dataloader import DataLoader


# %% Data loading     ##
######################
path = "coding/code/exchange_base/"

stage = "train"
input_file_name_vectorized = path + stage +  "_vectorized_1d.pt"
input_file_name_labels = path + stage +  "_labels_1d.pt"
vectors = torch.load(input_file_name_vectorized)
# Loading the label tensor
labels = torch.load(input_file_name_labels)

stage = "val"
input_file_name_vectorized_val = path + stage +  "_vectorized.pt"
input_file_name_labels_val = path + stage +  "_labels.pt"
vectors_val = torch.load(input_file_name_vectorized_val)
# Loading the label tensor
labels_val = torch.load(input_file_name_labels_val)

# %% Configuration
######################
# Batch Size for DataLoader
batch_size = 1
num_epochs = 5
num_classes = 3
learning_rate = 0.001

# %% Setup for CNN    ##
######################
# Output Input Data information
print("Matrix length: {:>5,}".format(len(vectors)))
print("labels length: {:>5,}".format(len(labels)))

# Combine Vectorizations with labels in TensorDataset
dataset = TensorDataset(vectors,labels)
# Setup PyTorch Dataloader
dataset_loader = DataLoader(dataset,
                #sampler = RandomSampler(dataset),
                batch_size = batch_size)


# Combine Vectorizations with labels in TensorDataset
dataset_val = TensorDataset(vectors_val,labels_val)
# Setup PyTorch Dataloader
dataset_loader_val = DataLoader(dataset_val,
                #sampler = RandomSampler(dataset),
                batch_size = batch_size)

classes = (0, 1, 2) #'hateful': '0', 'abusive': '1', 'normal': '2'

# %%
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

# #### Test
# # import single data point
# vec = vectors[0]
# label = labels[0]

# %%
# Convolutional neural network (two convolutional layers)
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(288, num_classes) #7*7*32
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN(num_classes).to(device)

# Output model graphviz
#graph = make_dot(model(tweetBertTensor.unsqueeze(0)), params=dict(model.named_parameters()))

# %%
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
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
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# %%
# Save model checkpoint
torch.save(model.state_dict(), path +"model" + "_epochs" + str(num_epochs) + ".ckpt")


# %%
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for tweetBertTensor, labels in dataset_loader_val:
        
        tweetBertTensor = tweetBertTensor.to(device)
        labels = labels.to(device)
        outputs = model(tweetBertTensor.unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test tweetBertTensor: {} %'.format(100 * correct / total))