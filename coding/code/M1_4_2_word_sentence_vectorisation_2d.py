# loading packages
import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn.functional as F

def vectorize(data, maxVectorLength=120, matrixColumns=10, matrixRows=12, textColumn="tweet", labelColumn="label", pretrainedModel="bert-base-uncased", verbose=True, randomSeed=42):
    """ Vectorizes each row of a specific dataframe column using a Bert pretrained model and outputs a two tensors.
    One containing the vectorized entries of the column and one containing the associated labels.

    Parameters:
        data (dataframe): The dataset from which the data should be extracted
        maxVectorLength (int): The length of the vector embedding. Will be padded with zeros if shorter
        matrixColumns (int): Vector embedding will be reshaped to matrix with this column amount
        matrixRows (int): Vector embedding will be reshaped to matrix with this row amount
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

    # Summary of the used transformers Package:
    # 🤗Transformers (formerly known as pytorch-transformers and 
    # pytorch-pretrained-bert) provides state-of-the-art general-purpose
    # architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, 
    # CTRL...) for Natural Language Understanding (NLU) and Natural
    # Language Generation (NLG) with over 32+ pretrained models in
    # 100+ languages and deep interoperability between TensorFlow 2.0
    # and PyTorch.

    # Load pretrained Tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(pretrainedModel)


    # Stats output initialization
    lengthSample = [] 
    # Embeddings 
    #vectorEmbeddings = torch.zeros([0,len(data), 2], dtype=torch.int32)
    vectorEmbeddings = torch.Tensor()

    for i,tweetText in enumerate(data[textColumn].values):
        # Encode tweet
        if(isinstance(tweetText, float)): #empty value is interpreted as nan
            print("Float tweet found in data "+str(i)+": \""+str(tweetText)+"\" --> interpreting it as string with str(tweet)")
        
        tweetText = str(tweetText) #empty tweets were interpreted as float 

        encoding = tokenizer.encode(tweetText, max_length=maxVectorLength)
        # Stats output
        lengthSample.append(len(encoding))
        # Add zeros at the end of the vector until maxVectorLength is reached
        paddedTensor = torch.zeros(1,maxVectorLength)
        paddedTensor[:,:len(encoding)] = torch.tensor(encoding) 
        # convert to matrix/tensor for CNN
        reshapedTensor = paddedTensor.view((matrixColumns, matrixRows))
        # without unsqueezing all the tensors will be concatenated into one big matrix instead of multiple small ones
        unequeezedTensor = reshapedTensor.unsqueeze(0)
        # save into tensor
        vectorEmbeddings = torch.cat((vectorEmbeddings,unequeezedTensor),dim=0)
        
        #print("#"+str(i)+" check: "+str(i % 1000))
        # Progress bar 
        if(i % 30 == 0):
            print("Progress: "+str(round(i/len(data),3)), end ="\r")


        # vectorEmbeddings.append(paddedEncodingMatrix)

    # Progress bar    
    print("")


    # convert Stats helper vector to DataFrame for stats function usage
    lengthSample = pd.DataFrame(lengthSample) 

    # Output stats information about tweet representation
    if(verbose):
        print("Tweets: "+str(len(vectorEmbeddings))+" mean-length: "+str(lengthSample.mean()[0])+" median-length: "+str(lengthSample.median()[0])+" min: "+str(lengthSample.min()[0])+" max: "+str(lengthSample.max()[0]))

    # Convert vector of vectors to tensor
    #matrix = torch.tensor(vectorEmbeddings, dtype=torch.float32)
    labels = torch.tensor(data[labelColumn].values)

    return vectorEmbeddings, labels


### Preparing input data
# File source possibilities (uncomment what applies to you)
# 1. download from github
# input_file = "https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/code/exchange_base/train_set.csv"
# output_file_name = "exchange_base/train_vec.pt"
# 2. use exchange_base files

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

if __name__ == "__main__":
    path = "exchange_base/"

    createTensors(path,"train")
    createTensors(path,"val")
    createTensors(path,"test")