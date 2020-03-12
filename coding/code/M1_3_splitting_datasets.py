
# importing packages
import pandas as pd

# splitting datasets into train, evaluation, and test set

url_data_cleaned = 'https://raw.githubusercontent.com/MaximilianKupi/nlp-project/master/coding/code/exchange_base/data.csv'

data_cleaned = pd.read_csv(url_data_cleaned)

data_cleaned.head()