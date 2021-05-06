import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

import acquire as a

#Prepare NLP Text

#function to clean
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
#function to clean
def basic_clean(string):
    """
    This function takes in one argument (string) and will apply
    some basic text cleaning to it:
    1. lowercase everything
    2. normalize unicode characters
    3. replace anything that is not a letter, number, whitespace,
    or a single quote
    """
    lowercase = string.lower()
    normalize = unicodedata.normalize('NFKD', lowercase)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    remove_special = re.sub(r"[^a-z0-9'\s]", '', normalize)
    clean_string = remove_special
    return clean_string

#function to tokenize
def tokenize(string):
    """
    This function will take in one argument(string) and will
    tokenize all words in the string.
    """
    #create tokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()
    #use tokenizer
    tokens = tokenizer.tokenize(string, return_str = True)
    
    return tokens
    

#function to stem
def stem(string):
    # Create porter stemmer.
    ps = nltk.porter.PorterStemmer()
    # Apply the stemmer to each word in our string.
    stems = [ps.stem(word) for word in string.split()]
    
    return stems

#function to lemmatize
def lemmatize(string):
    # Create the Lemmatizer.
    wnl = nltk.stem.WordNetLemmatizer()
    # Use the lemmatizer on each word in the list of words we created by using split.
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    # Join our list of words into a string again; assign to a variable to save changes.
    lemmatized_string = ' '.join(lemmas)
    
    return lemmas

#function to remove stop words
def remove_stopwords(string, extra_words=[], exclude_words=[]):
    """
    This function will take in three arguments string, extra_words,
    and exclude words.  
    """
    # Create stopword_list.
    stopword_list = stopwords.words('english')
    
    # Remove 'exclude_words' from stopword_list to keep these in my text.
    stopword_list = set(stopword_list) - set(exclude_words)
    
    # Add in 'extra_words' to stopword_list.
    stopword_list = stopword_list.union(set(extra_words))
    
    # Split words in string.
    words = string.split()
    
    # Create a list of words from my string with stopwords removed and assign to variable.
    filtered_words = [word for word in words if word not in stopword_list]
    
    # Join words in the list back into strings and assign to a variable.
    string_without_stopwords = ' '.join(filtered_words)
    
    return string_without_stopwords


#ravi's function
def prep_article_data(df, column, extra_words=[], exclude_words=[]):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    df['stemmed'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(stem)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    df['lemmatized'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(lemmatize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    return df[['title', column,'clean', 'stemmed', 'lemmatized']]

#matthews all in one
def clean_stem_stop(string):
    """
    This function will take in a string and 
    1. clean
    2. tokenize
    3. stem
    """
    return remove_stopwords(stem(tokenize(basic_clean(string))))

def clean_lem_stop(string):
    """
    This function will take in a string and
    1. clean
    2. tokenize
    3. lemmatize
    """
    return remove_stopwords(lemmatize(tokenize(basic_clean(string))))