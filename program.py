import os, re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
root=tk.Tk()
id_stem = StemmerFactory()
idn_stemmer = id_stem.create_stemmer()
directory=''
id_dir = 'ID/'
en_dir = 'EN/'

bhseng = open(id_dir+'bahasa inggris.txt', 'r', encoding='utf-8').read()
nato = open(id_dir+'nato.txt', 'r', encoding='utf-8').read()
rus = open(id_dir+'rusia.txt', 'r', encoding='utf-8').read()
orba = open(id_dir+'soeharto.txt', 'r', encoding='utf-8').read()
ww = open(id_dir+'perang dunia 2.txt', 'r', encoding='utf-8').read()
bhseng2 = open(en_dir+'english.txt', 'r', encoding='utf-8').read()
nato2 = open(en_dir+'nato.txt', 'r', encoding='utf-8').read()
rus2 = open(en_dir+'russia.txt', 'r', encoding='utf-8').read()
orba2 = open(en_dir+'suharto.txt', 'r', encoding='utf-8').read()
ww2 = open(en_dir+'ww2.txt', 'r', encoding='utf-8').read()

filelist = [bhseng,nato,rus,orba,ww]
filelist2 = [bhseng2,nato2,rus2,orba2,ww2]
tokenlist = []
tokenlist2 = []

for file in filelist:
    file = re.sub(r"[\W\s(0-9)]+", ' ', file.lower())
    file = idn_stemmer.stem(file)
    token = word_tokenize(file)
    tokenlist.append(token)
for file in filelist2:
    file = re.sub(r'[\W\s(0-9)]+', ' ', file.lower())
    token = word_tokenize(file)
    tokenlist2.append(token)

tokens_filtered = []
tokens_filtered2 = []

stop_words = set(stopwords.words('indonesian'))
for tokens in tokenlist:
    token_filtered = []
    for t in tokens:
        if t not in stop_words:
            token_filtered.append(t)
    tokens_filtered.append(token_filtered)
    
stop_words = set(stopwords.words('english'))
for tokens in tokenlist2:
    token_filtered = []
    for t in tokens:
        if t not in stop_words:
            token_filtered.append(t)
    tokens_filtered2.append(token_filtered)


corpus_id = []
for tokens in tokens_filtered:
    doc=""
    for token in tokens:
        doc = doc+token+" "
    corpus_id.append(doc)
    
corpus_en = []
for tokens in tokens_filtered2:
    doc=""
    for token in tokens:
        doc = doc+token+" "
    corpus_en.append(doc)

vec_id = TfidfVectorizer()
tfidf_wm = vec_id.fit_transform(corpus_id)
tfidf_term = vec_id.get_feature_names_out()
result_id = pd.DataFrame(
    data=tfidf_wm.toarray(), 
    index=["Bhs Ing", "NATO", "Rusia", "Soeharto","PDII"], 
    columns=tfidf_term)

print(result_id)

vec_en = TfidfVectorizer()
tfidf_wm2 = vec_en.fit_transform(corpus_en)
tfidf_term2 = vec_en.get_feature_names_out()
result_en = pd.DataFrame(
    data=tfidf_wm2.toarray(), 
    index=["English", "NATO", "Russia", "Soeharto","WWII"], 
    columns=tfidf_term2)

# print()
print(result_en)
