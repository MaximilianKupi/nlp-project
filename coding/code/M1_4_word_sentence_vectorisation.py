# loading packages
import pandas as pd
import numpy as np
import transformers
import torch


### Settings

# Maximal length of vector representation of tweet
maxVectorLength = 10 #<=512

# setting seed for reproducability
np.random.seed(42)

# File source possibilities (uncomment what applies to you)
# 1. download from github
# input_file = "https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/data/twitter%20data/davidson%20et%20al./labeled_data.csv"
# output_file_name = "exchange_base/train_vec.pt"
# 2. use exchange_base files
path = "exchange_base/"
stage = "train"
input_file = path + stage + "_set.csv"
output_file_name = path + stage +  "_vec.pt"


### Algorithm

# loading data
data = pd.read_csv(input_file)


# 🤗Transformers (formerly known as pytorch-transformers and 
# pytorch-pretrained-bert) provides state-of-the-art general-purpose
# architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, 
# CTRL...) for Natural Language Understanding (NLU) and Natural
# Language Generation (NLG) with over 32+ pretrained models in
# 100+ languages and deep interoperability between TensorFlow 2.0
# and PyTorch.

# Load pretrained Tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

# Load pretrained Transformer
bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')

data1 = data['tweet'][1:5]

# Stats output initialization
lengthSample = [] 
# Embeddings 
vectorEmbeddings = []

for tweet in data['tweet'].values:
    # Encode tweet
    encoding = tokenizer.encode(tweet, max_length=maxVectorLength)
    # Stats output
    lengthSample.append(len(encoding))
    # Padding with zeros and adding to output vector
    amountZeros = maxVectorLength - len(encoding)
    # Add zeros at the end of the vector until maxVectorLength is reached
    vectorEmbeddings.append(np.pad(encoding, (0,amountZeros), 'constant'))

# convert Stats helper vector to DataFrame for stats function usage
lengthSample = pd.DataFrame(lengthSample) 
# Output stats information about tweet representation
print("Tweets: "+str(len(vectorEmbeddings))+" mean-length: "+str(lengthSample.mean()[0])+" median-length: "+str(lengthSample.median()[0])+" min: "+str(lengthSample.min()[0])+" max: "+str(lengthSample.max()[0]))

# Convert vector of vectors to tensor
matrix = torch.tensor(vectorEmbeddings, dtype=torch.float32)

# Saving Tensor to file
# Use torch.save(tensor, 'file.pt') and torch.load('file.pt') to save Tensors to file
torch.save(matrix,output_file_name)