#!/usr/bin/env python
# coding: utf-8


# -*- coding:utf-8 -*-

import re
import string
import random
import numpy as np

log_interval = 200
wordsList = np.load('wordsList.npy')
print('Loaded the word list!')

stopwords_list = 'a about after all also always am an and any are at be been being but by came can come could did do does doing else for from get give goes going had happen has have having how i if ill im in into is it its just keep let like made make many may me mean more most much no not now of only or our really say see some something take tell than that the their them then they thing this to try up us use used uses very want was way we what when where which who why will with without wont you your youre'
stopwords = stopwords_list.split()
# stopwords_list
wordsList = wordsList.tolist() #Originally loaded as numpy array
# wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
print(wordsList[0])
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')
print(len(wordsList))
print(wordVectors.shape)

from nltk.corpus import twitter_samples
from nltk.tag import pos_tag, pos_tag_sents
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')  
print(len(positive_tweets))

lemmatizer = WordNetLemmatizer()
non_word = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")


def clean_sentences(each_tweet):
    new_text = []
    new_tweet=''
    state_list = pos_tag(each_tweet)
    for word, state in state_list:
        # remove trends
        if word.startswith('#'):
            word = ''
        # remove website urls
        word = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:[0-9a-fA-F][0-9a-fA-F]))+', '', word)
        # remove userids
        word = re.sub('(@[A-Za-z0-9_]+)', '', word)
        # remove numerical words
        word = re.sub(non_word, '', word)
        
        if state.startswith('NN'):
            pos = 'n'
        elif state.startswith('VB'):
            pos = 'v'
        else:
            pos='a'
        new_word = lemmatizer.lemmatize(word,pos)
        if len(new_word) > 0 and new_word not in string.punctuation and new_word.lower() not in stopwords:
            new_text.append(new_word.lower()) 
        new_tweet+=' '+new_word.lower()
    return new_text, new_tweet 

# s = positive_tweets[0].split()
# print(s)
# print(clean_sentences(s))



positive_cleaned_list = [clean_sentences(tweet.split()) for tweet in positive_tweets]
print(positive_cleaned_list[0])
len(positive_cleaned_list)

negative_cleaned_list = [clean_sentences(tweet.split()) for tweet in negative_tweets]
print(negative_cleaned_list[0])
len(negative_cleaned_list)

tokenset  = {}
cleaned_list = positive_cleaned_list + negative_cleaned_list
cleaned_tweets = [tweet[1] for tweet in cleaned_list]
cleaned_list = [tweet[0] for tweet in cleaned_list]
label = np.asarray([1]*5000+[0]*5000)
for words in cleaned_list:
    for word in words:
        if word in tokenset:
            tokenset[word]+=1
        else:
            tokenset[word]=1

tokenset_keys = [key for key in tokenset]
tokenset_keys=np.array(tokenset_keys)
np.save('tokens', tokenset_keys)


def tfvec_gen(words, tokenset_keys,  smoothing = False):
    dct = dict.fromkeys(tokenset_keys, 0)
    for word in words:
        if word in tokenset_keys:
            dct[word] +=1
    vec = [dct[token]for token in tokenset_keys]
    # print(vec)
    if smoothing: 
        text_words = len(words)
        total_tokens = len(tokenset)
        vec = [(x+1)/(text_words+total_tokens) for x in vec]
    return vec
    
def tfvecs_gen(wordslists_2d, tokenset, smoothing = False):
    l = len(wordslists_2d)
    keys = [token[0] for token in tokenset]
    tfvecs = np.zeros((l,len(tokenset)))
    for i in range(l):
        tfvecs[i] = tfvec_gen(wordslists_2d[i], keys, smoothing)
        if not i%log_interval:
            print('dealing with (',i,'/',l,') text file')
    return tfvecs   

print('initializing tf_vecs ...')
tfvecs = tfvecs_gen(cleaned_list, tokenset)
print('tf_vecs generated')

np.save('tfvecMatrix', tfvecs)
np.save('yLabel',label) 
newmatrix = np.hstack((np.array(cleaned_tweets).reshape(10000,1), np.array(label).reshape(10000,1)))
print(newmatrix.shape)
np.savetxt('newMatrix.csv', newmatrix, delimiter=',',fmt='%s',encoding='utf-8')
print('done')



