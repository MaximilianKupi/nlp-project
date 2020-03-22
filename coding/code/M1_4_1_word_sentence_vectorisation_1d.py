# %%
# loading packages
import pandas as pd
import numpy as np
import transformers
import torch
import tqdm

# %%
def vectorize(data, maxVectorLength=120, textColumn="tweet", labelColumn="label", pretrainedModel="bert-base-uncased", verbose=True, randomSeed=42):
    """ Vectorizes each row of a specific dataframe column using a Bert pretrained model and outputs a two tensors.
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
    vectorEmbeddings = []

    tweetAmount = data["tweet"].values.size

    for i,tweet in enumerate(data[textColumn].values):
        # Encode tweet
        encoding = tokenizer.encode(tweet, max_length=maxVectorLength)
        # Stats output
        lengthSample.append(len(encoding))
        # Padding with zeros and adding to output vector
        amountZeros = maxVectorLength - len(encoding)
        # Add zeros at the end of the vector until maxVectorLength is reached
        vectorEmbeddings.append(np.pad(encoding, (0,amountZeros), 'constant'))

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

if __name__ == "__main__":
    ### Preparing input data
    # File source possibilities (uncomment what applies to you)
    # 1. download from github
    # input_file = "https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/code/exchange_base/train_set.csv"
    # output_file_name = "exchange_base/train_vec.pt"
    # 2. use exchange_base files
    path = "coding/code/exchange_base/"
    stage = "train"
    input_file = path + stage + "_set.csv"
    output_file_name_vectorized = path + stage +  "_vectorized_1d.pt"
    output_file_name_labels = path + stage +  "_labels_1d.pt"

    # loading data
    data = pd.read_csv(input_file)

    ### Executing the function
    matrix, labels = vectorize(data)

    ### Saving the data
    # Saving the vectorized tweets tensor
    torch.save(matrix,output_file_name_vectorized)
    # Saving the label tensor
    torch.save(labels,output_file_name_labels)

    # Use torch.save(tensor, 'file.pt') and torch.load('file.pt') to save Tensors to file