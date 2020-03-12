import pandas as pd
import re

# MODULE 1: Data preprocessing and loading

# (a) Obtaining and cleaning the datasets
data_davidson = pd.read_csv('/Users/mxm/Google Drive/Masterstudium/Inhalte/4th Semester/NLP/nlp-project/coding/data/twitter data/davidson et al./labeled_data.csv',  engine='python')
data_founta = pd.read_csv('/Users/mxm/Google Drive/Masterstudium/Inhalte/4th Semester/NLP/nlp-project/coding/data/twitter data/founta et al./hatespeech_text_label_vote.csv', sep='\t', names=["tweet", "Majority Label", "Votes"], engine='python')

data_founta = data_founda[~data_founta.Label.str.contains("spam")]
#data_founta['class'] = data.founta['Majority Label'].replace({'abusive': '1', 'normal': '2', 'c': 'w', 'd': 'z'})



#data_founta['class'].map()
#s.map({'cat': 'kitten', 'dog': 'puppy'})


# (b) Twitter specific text pre-processing

#def  clean_tweets(df, text_field):
 #   df[text_field] = df[text_field].str.lower()
  #  df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
   # return dftest_clean = clean_text(test, "tweet")
#train_clean = clean_text(train, "tweet")

def RemoveSmallWords(text)
    frase = []
    for word in text.split():
        if len(word) > 3:
            frase.append(word)
    frase = " ".join(frase)
    return frase


# (c) Word / sentence vectorisation (as first data input)

# Michael: BERT and FastText

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