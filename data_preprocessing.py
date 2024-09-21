# -*- coding: utf-8 -*-
"""
Created on Mar  5 16:46:09 2024

@author: N Ouben
"""

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd 
from nltk.stem import WordNetLemmatizer
import json

nltk.download('stopwords')
nltk.download('punkt')



# URL cleaning function

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+|http?://\S+')
    return url.sub(r'', text)

# Emoji removal function
def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  
        u'\U0001F300-\U0001F5FF'  
        u'\U0001F680-\U0001F6FF'  
        u'\U0001F1E0-\U0001F1FF'  
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Punctuation removal function
def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

# Newlines removal function
def remove_newlines(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Numbers removal function
def remove_digits(text):
  digit = re.compile(r'[0-9]*')
  return digit.sub(r'', text)

# Repetitive letters removal
def remove_repetitive_letters(text):
    pattern = r'(\w)\1{2,}'  
    cleaned_text = re.sub(pattern, r'\1', text)
    return cleaned_text



# Load the contractions dictionary from the JSON file
with open('contractions.json', 'r') as json_file:
    contractions_dict = json.load(json_file)

# Compile the regular expression for contractions
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


# Function to expand contractions based on contractions dictionnary
def expand_contractions(text,contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, text)


# stopwords removal
def remove_stop_words(text) :

  # Lowercasing
  text = text.lower()
  

  # tokenization
  tokens = word_tokenize(text)

  stop_words = set(stopwords.words('english')) 
  tokens = [token for token in tokens if token not in stop_words]

  return(' '.join(tokens))


# Lemmatization (optional step)
def lemmatize(text) :

  tokens = word_tokenize(text)
  lemmatizer = WordNetLemmatizer()
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

  return(' '.join(lemmatized_tokens))





#######################################################################################################

file_path = "../data/metfo_insu_gd_posts.csv" 


  # Reading the CSV file
df = pd.read_csv(file_path, 
                 delimiter=';', 
                 encoding='utf-8')


  # Applying the preprocessing functions on the text data
df['text'] = df['text'].apply(lambda x: remove_URL(x))  # retrait des url
df['text'] = df['text'].apply(lambda x: remove_emoji(x)) # retrait des emoji
df['text'] = df['text'].apply(lambda x: remove_digits(x)) # retrait des chiffres
df['text'] = df['text'].apply(lambda x: remove_repetitive_letters(x)) # retrait des lettres rÃ©pÃ©titives
df['text'] = df['text'].apply(lambda x: expand_contractions(x)) # expansion contraction et abreviations
df['text'] = df['text'].apply(lambda x: remove_punct(x)) # retrait ponctuation
df['text'] = df['text'].apply(lambda x: remove_stop_words(x)) # retrait des mots blancs

  # lemmatisation 
df['text'] = df['text'].apply(lambda x : lemmatize(x))



# Exporting the preprocessed posts

df.to_csv("../data/cleaned_posts.csv", 
          encoding = "utf-8", 
          index = False, 
          sep = ";", 
          decimal = ",")





