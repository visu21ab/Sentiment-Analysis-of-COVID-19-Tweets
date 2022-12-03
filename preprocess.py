import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

def strftime_data(data, strformat: str):
    return data\
           .dt.strftime(strformat)\
           .value_counts()\
           .sort_index()\
           .reset_index(name='counts')

def _clean_tweets(df):
    ''' Removing unncessary features in the tweet texts.'''

    #removing url's, user tags, special characters except punctuations, 
    regex_list = {
        # email
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+': '',
        # user mentions
        r'(@|#)\w+': '',
        # non word characters
        r'\W': ' ',
        # orphans
        r'\b\w{1}\b': '',
        # pure numbers
        r'\b\d+\.?\d?\b': '',
        # remove amp
        r'amp': '',
        # remove whitespaces
        r'\s{2,}': ' '
    }

    for reg, replace in regex_list.items():
        df.Tweet = df.Tweet.str.replace(reg, replace) 

    #capitalisation
    df.Tweet = df.Tweet.str.lower()
    # return df

def _concentrate_tweets(df):
    ''' remove frequent words without contextual meaning and word variations'''
    #remove stop-words
    stop_words = stopwords.words('english')
    df.Tweet = df.Tweet.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    #lemmatization
    lemmatizer = WordNetLemmatizer()
    df.Tweet = df.Tweet.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])) 

def tweets(df):
    ''' Calling preprocessing steps to handle tweets before it is fed into model. '''
    preprocessed_df = df.copy()
  
    _clean_tweets(preprocessed_df) 
    _concentrate_tweets(preprocessed_df)
    preprocessed_df['tokenized_tweets'] = preprocessed_df.Tweet.apply(lambda x: x.split())

    return preprocessed_df

def word_count(items):
    """Calculates words in a sentence for each df item"""
    text_len = np.empty(len(items))
    for i, text in enumerate(items):
        text_len[i] = len(text.split())
    return text_len