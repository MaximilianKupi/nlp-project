import pandas as pd

# MODULE 1: Data preprocessing and loading

# (a) Obtaining and cleaning the datasets
data_davidson = pd.read_csv('/Users/mxm/Google Drive/Masterstudium/Inhalte/4th Semester/NLP/nlp-project/coding/data/twitter data/davidson et al./labeled_data.csv',  engine='python')
data_founta = pd.read_csv('/Users/mxm/Google Drive/Masterstudium/Inhalte/4th Semester/NLP/nlp-project/coding/data/twitter data/founta et al./hatespeech_text_label_vote.csv', sep='\t', names=["tweet", "Majority Label", "Votes"], engine='python')
print(data_davidson.head())
print(data_founta.head())



# (b) Twitter specific text pre-processing
# (c) Word / sentence vectorisation (as first data input)
# (d) Implementing a dictionary approach potentially based on Hatebase.org (as second data input)
# (e) Splitting data into test, validation and testing set
# (f) Specifying and implementing the data loader
# (g) Testing and iterating the module
# (h) Creatingmodule-specificvisualisationsforthefinal paper

















# MODULE 2: Model architecture and training

# (a) Choosing width and depth of the model
# (b) Choosingoptimizeraswellasactivationandloss functions
# (c) Choosing stopping rule, regularisation, dropout, learning rates etc.
# (d) Potentially performing a hyperparameter grid search
# (e) Running and tracking the training
# (f) Training and tuning the model based on the re- sults of the validation set
# (g) Creatingmodule-specificvisualisationsforthefi- nal paper