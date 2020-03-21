# splitting datasets into train, evaluation, and test set

def split_data(data_cleaned=None, train_p=0.7, val_p=0.15, test_p=0.15, random_state=42, y='label'):
    """This function gets the cleaned dataset and splits it into train, validation and test set based on scikit learn's StratifiedShuffleSplit.

    The input dataframe can be specified with the argument "data_cleaned". If nothing is specified, the function will get the data from our github repository. 

    Also you can specify proportions for the split with "train_p" (default set to 0.7), "val_p" (default set to 0.15), and "test_p" (default set to 0.15).

    Lastly, 'y' (default to 'label') sets the the column in the dataframe, where the labels are stored and which will be used as reference for the stratified sampling method. 

    The function returns the training, validation, and test set (in that order). To assign them put "train_set, val_set, test_set = split_data()". 
    """
    # importing packages
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    # checking if val_p + test_p + train_p = 1 and throwing an error otherwise
    x = val_p + test_p + train_p
    if x != 1:
        raise Exception('train_p + val_p + test_p should sum up to 1, however they sum up to: {}'.format(x))

    # setting the github url from where to get the data
    url_data_cleaned = 'https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/code/exchange_base/data.csv'

    # reading the data in case it is not here already (since it will later all be applied in one pipeline)
    if data_cleaned is None:
        data_cleaned = pd.read_csv(url_data_cleaned, index_col='id')
        print('INFO: Reading data_cleaned anew from github since no input was provided')

    # instatiating the class for the shuffle split of training and test/validation set
    split = StratifiedShuffleSplit(n_splits=10, test_size = test_p, random_state = random_state)

    for train_val_index, test_index in split.split(data_cleaned, data_cleaned['label']):
        train_val_set = data_cleaned.loc[train_val_index]
        test_set = data_cleaned.loc[test_index]

    # instantiating the class for the shufle split of test and validation set
    # getting the new proportion for the validation set if train + val = 1
    val_p_sub = val_p * 20/17

    split2 = StratifiedShuffleSplit(n_splits=10, test_size = val_p_sub, random_state = random_state)

    for train_index, val_index in split2.split(train_val_set, train_val_set[y]):
        train_set = data_cleaned.loc[train_index]
        val_set = data_cleaned.loc[val_index]

    return train_set, val_set, test_set


# assigning the returned dataframes to variables
train_set, val_set, test_set = split_data()

# saving the dataframes
train_set.to_csv("/Users/mxm/Google Drive/Masterstudium/Inhalte/4th Semester/NLP/nlp-project/coding/code/exchange_base/train_set.csv")
val_set.to_csv("/Users/mxm/Google Drive/Masterstudium/Inhalte/4th Semester/NLP/nlp-project/coding/code/exchange_base/val_set.csv")
test_set.to_csv("/Users/mxm/Google Drive/Masterstudium/Inhalte/4th Semester/NLP/nlp-project/coding/code/exchange_base/test_set.csv")