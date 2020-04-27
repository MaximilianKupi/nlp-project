"""This is the script to define the vectorisation function for the 2D model with dictionary approach.
"""

# loading packages
import pandas as pd
import numpy as np
import transformers
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from M1_5_dictionary_approach_tweetlevel import hatesearch
from math import sqrt
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
from tqdm import tqdm
import timeit

import sys, os

def vectorize(data, maxVectorLength=120, textColumn="tweet", labelColumn="label", pretrainedModel="bert-base-uncased", verbose=False, randomSeed=42):
    """ Vectorizes each row of a specific dataframe column using a Bert pretrained model and our dictionary approach and outputs two tensors.
    One containing the embedded entries of the column and one containing the associated labels.

    Args:
        data (dataframe): The dataset from which the data should be extracted.
        maxVectorLength (int): The length of the vector embedding. Will be padded with zeros if shorter. Default: 120.
        textColumn (str): The name of the column in the dataframe data containing the tweets. Default: 'tweet'.
        labelColumn (str): The name of the column in the dataframe data contianing the labels. Default: 'label'.
        pretrainedModel (str): The name of the model to be used from the transformers package. Default: 'bert-based-uncased'.
        verbose (bool): Whether or not the script should output stats information about the input data: average, median, max, min of word count in tweets. Default: False.
        randomSeed (int): Random seed to ensure that results are reproducable. Default: 42. 

    Returns:
        embedings (torch tensor): The embedded texts from the textcolumn of the dataframe data.
        labels (torch tensors): The label vector for the associated embedded column.  
    """
    startTime = datetime.now()
    ### Settings
    # setting seed for reproducability
    np.random.seed(randomSeed)

   
    # Load pretrained Tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(pretrainedModel)

    embeddings = torch.zeros([len(data[textColumn]),2,120])

    num_cores = multiprocessing.cpu_count()

    hatebase_dic = None #pd.read_csv('coding/data/dictionary/hatebase/full_dictionary.csv', index_col = 'vocabulary_id')

    list = Parallel(n_jobs=num_cores, prefer="threads")(delayed(createMatrix)(tweet,tokenizer,maxVectorLength,pretrainedModel,hatebase_dic) for tweet in tqdm(data[textColumn]))

    embeddings = torch.stack(list).squeeze(1)
    print("saved tensor "+str(embeddings.size()))

    if(verbose):
        print("Tweets: "+str(len(embeddings)))
        print("First tweet: "+str(embeddings[0]))

    labels = torch.tensor(data[labelColumn].values)

    print(datetime.now() - startTime)
    return embeddings, labels

def createMatrix(tweetText,tokenizer,maxVectorLength,pretrainedModel,hatebase_dic):
    """Creates a matrix of word embeddings based on the embeddings of BERT and the dictionary approach.

    Args: 
        tweetText (str): The tweet as string.
        tokenizer (object): The instantiated tokenizer of the specific pretrained BERT model.
        maxVectorLength (int): The specified maximum vector length of the embeddings. 
        pretrainedModel (str): To specify which pretrained Bert model to use.
        hatebase_dic (dataframe): The hatebase dictionary as pandas dataframe.
    
    Returns: 
        matrix (torch tensor): The twodimensional matrix with BERT embeddings on the first and our dictionary approach on the second dimension. 
    """
    if(isinstance(tweetText, float)): #empty value is interpreted as nan
            print("Float tweet found in data: \""+str(tweetText)+"\" --> interpreting it as string with str(tweet)")
        
    tweetText = str(tweetText) #empty tweets were interpreted as float 
    
    raw_encoding = torch.tensor(tokenizer.encode(tweetText, max_length=maxVectorLength))
    
    vlength = raw_encoding.size()[0]

    encoding = padWithZeros(raw_encoding,maxVectorLength)

    hateMetric = padWithZeros(stretch(hatesearch(tweetText,hatebase_dic),vlength),maxVectorLength)

    matrix = torch.cat((encoding,hateMetric),0).unsqueeze(0)


    # embeddings[i] = matrix
    return matrix

    

### Preparing input data
# File source possibilities (uncomment what applies to you)
# 1. download from github
# input_file = "https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/code/exchange_base/train_set.csv"
# output_file_name = "exchange_base/train_vec.pt"
# 2. use exchange_base files
def padWithZeros(vector,n):
    """Adds zeros to the end of a vector until a certain size is reached.

    Args:
        vector (tensor): The input vector that should be padded
        n (int): The length of the output vector after padding

    Returns:
        tensor: padded vector

    """
    #print('before padding: ', vector)
    paddedTensor = torch.zeros(1,n)
    paddedTensor[:,:len(vector)] = vector
    return paddedTensor

def createTensors(path,stage):
    """ Opens the file:
    - path + stage + "_set.csv"
    vectorizes the data and saves the labels separate as
    - path + stage +  "_vectorized.pt"
    - path + stage +  "_labels.pt"

    Args:
        path (str): the path wheere the dataset is and the tensors will be saved to
        stage (str): prefix of the file

    """
    input_file = path + stage + "_set.csv"
    output_file_name_vectorized = path + stage +  "_vectorized_2d.pt"
    output_file_name_labels = path + stage +  "_labels_2d.pt"
    
    # loading data
    data = pd.read_csv(input_file)

    ### Executing the function
    matrix, labels = vectorize(data, verbose=True)

    ### Saving the data
    # Saving the vectorized tweets tensor
    torch.save(matrix,output_file_name_vectorized)
    # Saving the label tensor
    torch.save(labels,output_file_name_labels)

    # Use torch.save(tensor, 'file.pt') and torch.load('file.pt') to save Tensors to file

def stretch(vector,n,plot=False):
    """Stretches Vectors to desired length by interpolating values.

    Args:
        vector (array(float)): The input vector that should be stretched
        n (int): The length of the output vector after stretching
        plot (bool): Whether or not to plot the vectors.
    
    Returns:
        tensor: The stretched vector.

    """
    #print('before stretching: ', vector)
    
    # removing the start and stop tokens
    n = n - 2
    if(len(vector) == 0):
        return torch.zeros(n)

    if(n<len(vector)):
        return vector
        #raise ValueError("Shrinking not allowed")
    # Converts input to numpy array
    oldY = np.array(vector)
    # Treates input as y-values of a function on x-values created here
    oldX = np.linspace(0,len(vector)-1,len(vector))

    # Determine non-zero values that we have to keep
    # If we deviate from these points in the interpolation
    # then the result will shrink the dictionary evaluation number
    keep = np.where(oldY != 0)[0]

    # Create new vector with desired length
    newX = np.linspace(0,len(vector)-1,n)

    # Make sure that the nonzero values are kept as is after interpolation
    # by replacing the closest values with the ones we saved in keep
    for value in keep:
        newX[np.abs(np.array(newX) - value).argmin()] = value

    # interpolate to new 
    newY = np.interp(newX,oldX,oldY)

    # inserting zero as first element in vector, so that the hatevector only starts where the words start in the bert vector
    newY = np.insert(newY, 0, 0, axis=0)

    if(plot):
        plt.plot(oldX,oldY)
        plt.plot(newX,newY)

    return torch.tensor(newY)

if __name__ == "__main__":
    path = "coding/code/exchange_base/"

    createTensors(path,"train")
    createTensors(path,"val")
    createTensors(path,"test")