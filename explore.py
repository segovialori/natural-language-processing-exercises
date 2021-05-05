import re
import unicodedata
import pandas as pd
import nltk #natural language tool kit
from wordcloud import WordCloud

from env import user, password, host

#Functions for NLP Exploration

#acquire
def get_db_url(database, host=host, user=user, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'

#prepare
ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt'] #ltgt is html artifact

def clean(text):
    'A simple function to cleanup text data'
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

#explore
def show_counts_and_ratios(df, column):
    labels = pd.concat([df[column].value_counts(),
                    df[column].value_counts(normalize=True)], axis=1)
    labels.columns = ['n', 'percent']
    
    return labels

#frequency bar plot
def show_freqent_ngram(list_of_strings, n):
    top_20_ngrams = (pd.Series(nltk.ngrams(list_of_strings, n))
                      .value_counts()
                      .head(20))
    top_20_ngrams.sort_values().plot.barh(color='pink', width=.9, figsize=(10, 6))

    plt.title(f'20 Most frequently occuring {n} -grams')
    plt.ylabel(f'{n}-gram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ngrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1]) #broken for now will fix later
    _ = plt.yticks(ticks, labels)
    plt.show()