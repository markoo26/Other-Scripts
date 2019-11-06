## cd "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\Scripts"
##!# Regular expressions

####!# NLP ####!#
import webbrowser 
webbrowser.open_new_tab('https://towardsdatascience.com/gentle-start-to-natural-language-processing-using-python-6e46c07addf3')
webbrowser.open_new_tab('https://dzone.com/articles/nlp-tutorial-using-python-nltk-simple-examples')
webbrowser.open_new_tab('https://www.guru99.com/tagging-problem-nltk.html')
webbrowser.open_new_tab('https://www.guru99.com/counting-pos-tags-nltk.html')
webbrowser.open_new_tab('https://www.nltk.org/book/ch05.html') ### POS-Tags
webbrowser.open_new_tab('https://www.guru99.com/word-embedding-word2vec.html')
webbrowser.open_new_tab('https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10')
webbrowser.open_new_tab('https://www.digitalocean.com/community/tutorials/how-to-work-with-language-data-in-python-3-using-the-natural-language-toolkit-nltk')
#!# Wybor instalowanych paczek do NLTK #!#
# cd "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\Scripts"

import os
os.environ['PATH'] += os.pathsep + 'C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\Scripts'
os.chdir('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\NLP')
##!# Biblioteki PyPDF2 -> import PDF

import nltk
#nltk.download() ##!# Do pobierania bibliotek w NLTK

####!# Grabbing data from a webpage ####!#

import urllib.request
response =  urllib.request.urlopen('https://en.wikipedia.org/wiki/Wisla_Krakow')
html = response.read()
print(html)

####!# Cleaning out data from XML/HTML tags ####!#

from bs4 import BeautifulSoup
soup = BeautifulSoup(html,'html5lib')
text = soup.get_text(strip = True)
print(text)

####!# Converting the HTML text into tokens ####!#

tokens = [t for t in text.split()]
print(tokens)

####!# Deleting the stopwords ####!#

from nltk.corpus import stopwords
sr= stopwords.words('english')
clean_tokens = tokens[:]
for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)
        
####!# Plotting the words distribution ####!#
        
freq = nltk.FreqDist(clean_tokens)
#for key,val in freq.items():
#    print(str(key) + ':' + str(val))
freq.plot(20, cumulative=False)

####!# Tokenizowanie tekstow ####!#

from nltk.tokenize import sent_tokenize 
mytext = "Hello Adam, how are you? I hope everything is going well. Today is a good day, see you dude." 
print(sent_tokenize(mytext))

####!# Niepoprawne tokenizowanie (branie kropki jako separator) ####!#

from nltk.tokenize import word_tokenize 
mytext = "Hello Mr. Adam, how are you? I hope everything is going well. Today is a good day, see you dude." 
print(word_tokenize(mytext))

####!# WordNet - znaczenie i synonimy slow ####!#

from nltk.corpus import wordnet
syn = wordnet.synsets("victory")

print(syn[0].definition())
print(syn[0].examples())

from nltk.corpus import wordnet 

synonyms = []

for syn in wordnet.synsets('beautiful'):

    for lemma in syn.lemmas():
        synonyms.append(lemma.name())

print(synonyms)

####!# Antonimy ####!#

antonyms = []

for syn in wordnet.synsets("beautiful"):
    for l in syn.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(antonyms)

####!# Stemming ####!#

from nltk.stem import PorterStemmer
stemmer = PorterStemmer() 
print(stemmer.stem('repeating'))

####!# Snowball - inny algorytm do stemmingu ####!#

from nltk.stem import SnowballStemmer
print(SnowballStemmer.languages)

from nltk.stem import SnowballStemmer
german_stemmer = SnowballStemmer('german')
print(german_stemmer.stem("verstehe"))


####!# Lemmatizing (lepsze, ale wolniejsze od stemmingu) ####!#

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('increases'))

## Przymiotnik/rzeczownik/czasownik

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('plays', pos="n")) ##!# Dostepne: v,n,a,r

####!# POS-Tagi ####!#

##!# Dwa glowne rodzaje tagow: 
##!# Rule based - dla slow o roznych znaczeniach, analizujemy slowa przed i po
##!# Stochastic POS - wchodzi w gre czestotliwosc i prawdopodobienstwo, gdy dane slowo
##!# ma najwiecej tagow X w train to w test dostaje ten sam tag

import nltk
text = "Python is very easy programming language to learn and offers good opportunities"
sentence = nltk.sent_tokenize(text)
for sent in sentence:
	 print(nltk.pos_tag(nltk.word_tokenize(sent)))


####!# Obliczanie ile POS (Part of Speech) - tagow jest w danym tekscie ####!#
     
from collections import Counter
text = "Python is very easy programming language to learn and offers good opportunities"
lower_case = text.lower()
tokens = nltk.word_tokenize(lower_case)
tags = nltk.pos_tag(tokens) ###!# Rodzaje POS-Tagow: https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
counts = Counter( tag for word,  tag in tags) ###!# Counter
print(counts)



####!# Obliczanie liczebnosci poszczegolnych slow ####!#

import nltk
a = "Guru99 is the site where you Beginners Beginners Beginners Beginners can find the best tutorials for Software Testing     Tutorial, SAP Course for Beginners. Java Tutorial for Beginners and much more. Please     visit the site guru99.com and much more."
words = nltk.tokenize.word_tokenize(a)
fd = nltk.FreqDist(words)
fd.plot()

####!# Bi-gramy ####!#

text = "Guru99 is a totally new kind of learning experience."
Tokens = nltk.word_tokenize(text)
output = list(nltk.bigrams(Tokens))
print(output)

####!# Trigramy ####!#

text = "Guru99 is a totally new kind of learning experience."
Tokens = nltk.word_tokenize(text)
output = list(nltk.trigrams(Tokens))
print(output)

####!# Count-Vectorizer ####!#

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
data_corpus=["guru99 is the best sitefor online online online tutorials. I love to visit guru99."]
vocabulary=vectorizer.fit(data_corpus)
X= vectorizer.transform(data_corpus)
print(X.toarray())
print(vocabulary.get_feature_names())

####!# Wprowadzenie do Gensim ####!#

import gensim
import time
from nltk.corpus import abc

start_time = time.time()
model= gensim.models.Word2Vec(abc.sents())
X= list(model.wv.vocab)
data=model.most_similar('science')
print(data)
end_time = time.time()
total_time = end_time - start_time
print(total_time/60) ##!# 16 minut

###!# Gensim is imported. If Gensim is not installed, please install it using the command " pip3 install gensim". Please see the below screenshot. 


####!# Budowanie inteligentnego chatbota ####!#

data = [{"tag": "welcome",
"patterns": ["Hi", "How are you", "Is any one to talk?", "Hello", "hi are you available"],
"responses": ["Hello, thanks for contacting us", "Good to see you here"," Hi there, how may I assist you?"]

        },
{"tag": "goodbye",
"patterns": ["Bye", "See you later", "Goodbye", "I will come back soon"],
"responses": ["See you later, thanks for visiting", "have a great day ahead", "Wish you Come back again soon."]
        },

{"tag": "thankful",
"patterns": ["Thanks for helping me", "Thank your guidance", "That's helpful and kind from you"],
"responses": ["Happy to help!", "Any time!", "My pleasure", "It is my duty to help you"]
        },
        {"tag": "hoursopening",
"patterns": ["What hours are you open?", "Tell your opening time?", "When are you open?", "Just your timing please"],
"responses": ["We're open every day 8am-7pm", "Our office hours are 8am-7pm every day", "We open office at 8 am and close at 7 pm"]
        },

{"tag": "payments",
"patterns": ["Can I pay using credit card?", " Can I pay using Mastercard?", " Can I pay using cash only?" ],
"responses": ["We accept VISA, Mastercard and credit card", "We accept credit card, debit cards and cash. Please don’t worry"]
        }
   ]

####!# Przyklad importowania pliku intents zamiast recznego podawania jak wyzej ####!#

#import json
#json_file =’intents.json'
#with open('intents.json','r') as f:
#    data = json.load(f)
    
####!# Przerabianie w PD DataFrame ####!#

import pandas as pd
df = pd.DataFrame(data)
df['patterns'] = df['patterns'].apply(', '.join) ##!# Przerobienie ze slownikow na string (?)

import string
####!# Preprocessing tekstow ####!#

from nltk.corpus import stopwords
from textblob import Word
stop = stopwords.words('english')
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x.lower() for x in x.split()))
df['patterns']= df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation))
df['patterns']= df['patterns'].str.replace('[^\w\s]','') ##!# Regular expressions
df['patterns']= df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x for x in x.split() if not x in stop))
df['patterns'] = df['patterns'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(df['patterns'])

####!# Word2VEC ####!#

from gensim.models import Word2Vec
Bigger_list=[]
for i in df['patterns']:
     li = list(i.split(" "))
     Bigger_list.append(li)	
Model= Word2Vec(Bigger_list,min_count=1,size=300,workers=4)

####!# Saving Model into output file ####!#

model.save("word2vec.model")
model.save("model.bin")

####!# Loading the previously built model ###!#

model = Word2Vec.load('model.bin')

vocab = list(model.wv.vocab) 
vocab

####!# Checking similar words ####!#

similar_words = model.most_similar('remove')	
print(similar_words)	

####!# Printing out dissimilar words ####!#

dissimlar_words = model.doesnt_match('See you later, thanks for visiting'.split())
print(dissimlar_words)

####!# Checking dissimilarity between two words ####!#

similarity_two_words = model.similarity('yes','no')
print("Please provide the similarity between these two words:")
print(similarity_two_words)

####!# Finding similar words ####!# 

similar = model.similar_by_word('remove')
print(similar)

####!# Seq-2-Seq model intro ####!#
#
#from __future__ import unicode_literals, print_function, division
#import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
#
#import numpy as np, pandas as pd
#import re, random
#
#device = torch.device("cuda" if  
#                     torch.cuda.is_available() else "cpu")

####!# Import tweetow z Twittera ####!#

from nltk.corpus import twitter_samples
from nltk.tag import pos_tag_sents

twitter_samples.fileids()
twitter_samples.strings('tweets.20150430-223406.json')
tweets = twitter_samples.strings('positive_tweets.json')
tweets_tokens = twitter_samples.tokenized('positive_tweets.json')
tweets_tagged = pos_tag_sents(tweets_tokens)

####!# Obliczanie tagow typu JJ oraz NN ####!#

JJ_count = 0
NN_count = 0
VB_count = 0

for tweet in tweets_tagged:
    for pair in tweet:
        tag = pair[1]
        if tag == 'JJ':
            JJ_count += 1
        elif tag == 'NN':
            NN_count += 1
        elif tag[:2] == 'VB':
            VB_count +=1
            
print('Total number of adjectives = ', JJ_count)
print('Total number of nouns = ', NN_count)
print('Total number of verbs = ', VB_count)


import os
import PyPDF2
os.chdir("C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia")

# creating an object 
file = open('thesoundcraftguidetomixing.pdf', 'rb')


## CIAG DALSZY
# creating a pdf reader object
fileReader = PyPDF2.PdfFileReader(file)

import textract
text = textract.process('thesoundcraftguidetomixing.pdf', method='pdfminer')
