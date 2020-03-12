# splitting datasets into train, evaluation, and test set

def split_data(data_cleaned=None, train=0.7, val=0.15, test=0.15):
    """This module gets the cleaned dataset and splits it into train, validation and test set.

    In a pipeline you can specify the input with the variable "data_cleaned". If nothing is specified, it will get the data from our github repository. 

    Also you can specify proportions for the split with "train" (default set to 0.7), "val" (default set to 0.15), and "test" (default set to 0.15).
    """
    # importing packages
    import pandas as pd
    import numpy as np
    import scikitlearn

    # setting seed for reproducability
    np.random.seed(42)

    # setting the github url from where to get the data
    url_data_cleaned = 'https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/code/exchange_base/data.csv'

    # reading the data in case it is not here already (since it will later all be applied in one pipeline)
    if data_cleaned:
        continue
    else:
        data_cleaned = pd.read_csv(url_data_cleaned)
        print('reading data_cleaned from github')


    data_cleaned.head()

