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
def basic_clean(text):
    """
    This function takes in one argument (text) and will apply
    some basic text cleaning to it:
    1. lowercase everything
    2. normalize unicode characters
    3. replace anything that is not a letter, number, whitespace,
    or a single quote
    """
    lowercase = text.lower()
    normalize = unicodedata.normalize('NFKD', lowercase)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    remove_special = re.sub(r"[^a-z0-9'\s]", '', normalize)
    clean_text = remove_special
    return clean_text

#function to tokenize
def tokenize(text):
    """
    This function will take in one argument(text) and will
    tokenize all words in the text.
    """
    tokenizer = nltk.tokenize.ToktokTokenizer()
    tokens = tokenizer.tokenize(basic_clean(text), return_str = True)
    
    return tokens

