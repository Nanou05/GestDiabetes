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




# English contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have", "gd" : "gestational diabetes"}

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

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





