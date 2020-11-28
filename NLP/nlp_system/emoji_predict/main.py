# encoding = utf-8
# 用于验证情感分析的正确性
api_use= False
import textblob
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np 
from gensim.models import Word2Vec
# from sklearn.externals 
import joblib
analyzer = SentimentIntensityAnalyzer()
w2v_model = Word2Vec.load('../data/w2v.model')
subjective_limit = 0.1
error = 0.05
maxlen  = 200
batch_size = 50

# 取得所有单词
vocab_list = list(w2v_model.wv.vocab.keys())
# 每个词语对应的索引
word_index = {word: index for index, word in enumerate(vocab_list)}

from keras.preprocessing.sequence import pad_sequences
# from nltk.corpus import twitter_samples
# from nltk.tag import pos_tag, pos_tag_sents
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk import FreqDist
from review_preprocessor import clean_review,get_index
from keras.models import load_model

if not api_use:
    # clf_svm = joblib.load('../models/svm.pkl')
    # clf_bys = joblib.load('../models/bys.pkl')
    clf_cnn = load_model('../models/cnn_model.hdf5')
    clf_lstm = load_model('../models/lstm_model.hdf5')

def sentiment_predict(sentence):
    tblob = TextBlob(sentence)
    vs = analyzer.polarity_scores(sentence)
    polarity = (tblob.polarity+vs['pos']-vs['neg'])/2
    if not api_use:
        cleaned = clean_review(sentence)
        vec_pad = word_vec(cleaned)
        x_input = np.zeros((batch_size,maxlen))
        x_input[0] = vec_pad
        # po1 = clf_svm.predict(vec_pad)
        # po2 = clf_bys.predict(vec_pad)
        po3 = clf_cnn.predict(x_input)
        print('cnn generated polarity = ',po3[0])
        po4 = clf_lstm.predict(x_input)
        print('lstm generated polarity = ', po4[0])
        # self_trained_po = 0.1*(po1+po2)+0.4*(po3+po4)
        self_trained_po = 0.5*(po3[0][0]+po4[0][0])
        print('api generated polarity = ',polarity)
        polarity = polarity/2+self_trained_po-0.5
    return polarity, tblob.subjectivity

def word_vec(cleaned_sentence):
    vec = get_index(cleaned_sentence, word_index)
    [vec_pad,] = pad_sequences([vec,],maxlen=maxlen)
    return vec_pad

import random
def getrightindex(buf, po_score):
    i = np.where(buflimits<po_score)
    # print(i)
    i=i[0][-1]
    # print(i)
    index_limit = int(buf[i][1])
    index_start = int(sum(buf[0:i-1][1]))if i>1 else 0
    return random.randint(index_start,index_limit+index_start)


def img_gen(sentence):
    po_score, sub_score = sentiment_predict(sentence)
    if sub_score>subjective_limit:
        if po_score>0.5:
            uni = "0x1f602"
        elif po_score>0:
            uni = "0x+1F60a"
        else:
            uni = "0x+1F622"
    else:
        uni = "0x+1F633"
    # 以下为较精细化的表情生成过程
    # 更正：因为网页显示原因，此部分已经停用。
    #     index = getrightindex(buf, po_score)
    #     # print(index)
    #     uni = emoji_list[index]['UNICODE']
    # else:
    #     index=random.randint(0, len(emoji_list))
    #     uni = emoji_list[index]['UNICODE']
    #     pass
        # np = nouns(sentence)
        
        # if np[0] in tag_dict:
        #     uni = tag_dict[np]
        # else:
        #     index = getrightindex(buf, po_score)
        #     uni = emoji_list[index]['UNICODE']
            
    return uni

def nouns(sentence):
    blob = TextBlob(sentence)
    np = blob.noun_phrases
    return np

'''
# # emoji_map[UNICODE] = {'name': NAME, 'tags': [TAG0, TAG1, TAG2...], 'sentiment': SENTIMENT}
# def tonp(emoji_list):
#     length = len(emoji_list)
#     emoji_np = np.ndarray((length,4))
#     for i in range(length):
#         emoji = emoji_list[i]
#         emoji_np[i][0]=emoji['UNICODE']
#         emoji_np[i][1]=emoji['name']
#         emoji_np[i][2]=emoji['sentiment']
#         emoji_np[i][3]=emoji['tags']
#     return emoji_np
'''

def buflist(emoji_list):
    l= len(emoji_list)
    buf = [[0,0]]
    now = 0
    last_sentiment=emoji_list[0]['sentiment']
    for i in range(1,(l//10)):
        buf[now][1]+=10
        if emoji_list[i*10]['sentiment']>last_sentiment+error:
            last_sentiment=emoji_list[i*10]['sentiment']
            buf.append([last_sentiment,0])
            now += 1
    return buf

def load_emoji(path):
    emoji_list = []
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        name_index = line.find('name:')
        sentiment_index = line.find('sentiment:')
        tags_index = line.find('tags:')
        unicodenum = line[10:name_index-2].split('\\')
        name = line[name_index+6: sentiment_index-2]
        if sentiment_index+1:
            sentiment = float(line[sentiment_index+11: tags_index-2])
        else:
            sentiment = 0
        tags = line[tags_index+6:].split()
        emoji = {'name':name, 'UNICODE':unicodenum, \
            'sentiment':sentiment, 'tags':tags}
        emoji_list.append(emoji)
    return emoji_list

def indexbytag(emoji_list):
    new_list = {}
    for emoji in emoji_list:
        tags = emoji['tags']
        for tag in tags:
            if tag in new_list:
                new_list[tag].append(emoji['UNICODE'])
            else:
                new_list[tag]=[emoji['UNICODE'],]
    # taglists = new_list.keys
    return new_list

if __name__=='__main__':
    emoji_list = load_emoji('emoji_map.txt')
    emoji_list.sort(key= lambda x:x['sentiment'])
    # emoji_np = np.array(emoji_list)
    buf = buflist(emoji_list)
    buflimits=np.array([buffer[1] for buffer in buf])
    tag_dict = indexbytag(emoji_list)
    # print(buf)
    # print(emoji_np[0])
    # print(emoji_list[-1])
    s = "it is nice weather today."
    polarity, subjectivity = sentiment_predict(s)
    print(polarity)
    print(img_gen(s))
