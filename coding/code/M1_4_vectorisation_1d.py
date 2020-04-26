"""This is the script to define the 1D vectorisation without our dictionary approach.
"""
# %%
# loading packages
import pandas as pd
import numpy as np
import transformers
import torch
import tqdm

# %%
def vectorize(data, maxVectorLength=120, textColumn="tweet", labelColumn="label", pretrainedModel="bert-base-uncased", verbose=True, randomSeed=42):
    """ Vectorizes each row of a specific dataframe column using a Bert pretrained model and outputs two tensors.
    One containing the vectorized entries of the column and one containing the associated labels.

    Parameters:
        data (dataframe): The dataset from which the data should be extracted
        textColumn (str): The name of the column in the dataframe data containing the tweets
        labelColumn (str): The name of the column in the dataframe data contianing the labels
        pretrainedModel (str): The name of the model to be used from the transformers package
        verbose (str): Whether or not the script should output stats information about the input data: average, median, max, min of word count in tweets
        randomSeed (str): Random seed to ensure that results are reproducable
        TODO Is random seed really necessary in this script?

    Returns:
        torch.tensor:The vectorized column of the dataframe data
        torch.tensor:The Label Vector for the associated vectorized column  

    """

    ### Settings
    # setting seed for reproducability
    np.random.seed(randomSeed)

    ### Algorithm


    # Load pretrained Tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(pretrainedModel)

    # Stats output initialization
    lengthSample = [] 
    # Embeddings 
    vectorEmbeddings = []

    tweetAmount = data["tweet"].values.size

    for i,tweet in enumerate(data[textColumn].values):
        tweet = str(tweet) #empty tweets were interpreted as float 
        # Encode tweet
        encoding = tokenizer.encode(tweet, max_length=maxVectorLength)
        # Stats output
        lengthSample.append(len(encoding))
        # Padding with zeros and adding to output vector
        amountZeros = maxVectorLength - len(encoding)
        # Add zeros at the end of the vector until maxVectorLength is reached
        vectorEmbeddings.append(np.pad(encoding, (0,amountZeros), 'constant'))

        if(i % 1000 == 0):
            print("Progress " + str(round(i/tweetAmount,3)))

    # convert Stats helper vector to DataFrame for stats function usage
    lengthSample = pd.DataFrame(lengthSample) 

    # Output stats information about tweet representation
    if(verbose):
        print("Tweets: "+str(len(vectorEmbeddings))+" mean-length: "+str(lengthSample.mean()[0])+" median-length: "+str(lengthSample.median()[0])+" min: "+str(lengthSample.min()[0])+" max: "+str(lengthSample.max()[0]))

    # Convert vector of vectors to tensor
    matrix = torch.tensor(vectorEmbeddings, dtype=torch.int)
    labels = torch.tensor(data[labelColumn].values, dtype=torch.int)


    return matrix, labels


def createTensors(path,stage):
    """ Opens the file:
    - path + stage + "_set.csv"
    vectorizes the data and saves the labels separate as
    - path + stage +  "_vectorized.pt"
    - path + stage +  "_labels.pt"

    Parameters:
        path (str): the path wheere the dataset is and the tensors will be saved to
        stage (str): prefix of the file

    """
    input_file = path + stage + "_set_wp.csv"
    output_file_name_vectorized = path + stage +  "_vectorized_1d_wp.pt"
    output_file_name_labels = path + stage +  "_labels_1d_wp.pt"
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

if __name__ == "__main__":
    path = "coding/code/exchange_base/"

    createTensors(path,"train")
    createTensors(path,"val")
    createTensors(path,"test")