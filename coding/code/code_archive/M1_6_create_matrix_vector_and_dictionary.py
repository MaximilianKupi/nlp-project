# %%
import torch
import torch.nn.functional as F
import numpy as np
from torch import squeeze

# %%
def merge(vectorized, hatefreq, hateint):
    """ merges a 1d vectorized representation of a tweet with the hatefrequency and 
    """


    length = vectorized.size()[1] # length of vectorization
    tweetsCount = vectorized.size()[0] # amount of tweets

    # %%
    torch.set_default_dtype(torch.float64)

    # %%
    for i in range(tweetsCount):
        
        # Create row for hatefreq
        # forces int conversion, assumes hatefreq is int
        hatefreqRow = torch.ones(length, dtype=torch.int32)*int(hatefreq[i].numpy().item())
        # Create row for hateint
        # forces int conversion, assumes hateint is int
        hateintRow = torch.ones(length, dtype=torch.int32)*int(hateint[i].numpy().item())
        tweet = vectorized[i]

        matrix = torch.stack([hatefreqRow, tweet, hateintRow])
        if (i == 0):
            # initialize tensor
            output = torch.unsqueeze(matrix,dim=0)
        else:
            # stack only works with initilized (in this case int) tensor
            output = torch.cat([output,torch.unsqueeze(matrix,dim=0)])
            #output = torch.cat([output, torch.unsqueeze(matrix,dim=0)],)
        
        if (i % 50 == 0):
            print("Progress "+str(round(i/tweetsCount,3)))

    return output

if __name__ == "__main__":
    # Running the function

    # %%
    # input of M1_4: vectorized tweets
    path = "coding/code/exchange_base/"
    stage = "train"
    vectorized = torch.load(path + stage +  "_vectorized_1d.pt")

    # input of M1_5: dictionary approach result
    hatefreq = torch.load(path +  "hatefreq.pt")
    hateint = torch.load(path + "hateint.pt")

    # merge tensors
    output = merge(vectorized, hatefreq, hateint)

    # save matrix tensor
    torch.save(output,path + stage +  "_vectorized_dict_context_2d.pt")