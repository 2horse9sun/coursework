from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np 
import re
import os 
analyzer = SentimentIntensityAnalyzer()
subjective_limit = 0.1
path = os.getcwd()+'\\'
api_use=True
error = 0.05
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag, pos_tag_sents
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
if not api_use:
    stopwords_list = 'a about after all also always am an and any are at be been being but by came can come could did do does doing else for from get give goes going had happen has have having how i if ill im in into is it its just keep let like made make many may me mean more most much no not now of only or our really say see some something take tell than that the their them then they thing this to try up us use used uses very want was way we what when where which who why will with without wont you your youre'
    stopwords = stopwords_list.split()
    lemmatizer = WordNetLemmatizer()
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
        # # remove numerical words
        # word = re.sub(non_word, '', word)
        
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

if not api_use:
    tokens = np.load(path+'tokens.npy')
    xvecs = np.load(path+'tfvecMatrix.npy')
    ylabels = np.load(path+'yLabel.npy')

def tfvec_gen(words, tokens,  smoothing = False):
    dct = dict.fromkeys(tokens, 0)
    for word in words:
        if word in tokens:
            dct[word] +=1
    vec = [dct[token]for token in tokens]
    # print(vec)
    if smoothing: 
        text_words = len(words)
        total_tokens = len(tokens)
        vec = [(x+1)/(text_words+total_tokens) for x in vec]
    return vec


    
if not api_use:
    from sklearn import naive_bayes
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    clf1 = naive_bayes.GaussianNB()
    clf1.fit(xvecs, ylabels)
    clf2 = naive_bayes.MultinomialNB()
    clf2.fit(xvecs, ylabels)
    #svm
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉，n_jobs=-1，多核计算
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    clf3 = grid.fit(xvecs, ylabels)


def sentiment_predict(sentence):
    if not api_use:
        words = clean_sentences(sentence)[0]
        xvec = tfvec_gen(words, tokens)
        tblob = TextBlob(sentence)
        vs = analyzer.polarity_scores(sentence)
        clf1_result = clf1.predict(xvec)
        clf2_result = clf2.predict(xvec)
        print(clf1_result,clf1_result)
    tblob = TextBlob(sentence)
    vs = analyzer.polarity_scores(sentence)
    polarity = (tblob.polarity+vs['pos']-vs['neg'])/2
    return polarity, tblob.subjectivity

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
        if po_score>0:
            uni = "&#x1f60a;"
        else:
            uni = "&#x1f622;"
    else:
        uni = "&#x1F633;"
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


def predict_emoji(input_str):
    emoji_list = load_emoji('./emoji_predict/emoji_map.txt')
    emoji_list.sort(key= lambda x:x['sentiment'])
    # emoji_np = np.array(emoji_list)
    buf = buflist(emoji_list)
    buflimits=np.array([buffer[1] for buffer in buf])
    tag_dict = indexbytag(emoji_list)
    # print(buf)
    # print(emoji_np[0])
    # print(emoji_list[-1])
    polarity, subjectivity = sentiment_predict(input_str)
    emoji = img_gen(input_str)
    return {"emoji": emoji, "polarity": polarity, "subjectivity": subjectivity}

if __name__=='__main__':
    print(predict_emoji("angry"))
    emoji_list = load_emoji('emoji_map.txt')
    emoji_list.sort(key= lambda x:x['sentiment'])
    # emoji_np = np.array(emoji_list)
    buf = buflist(emoji_list)
    buflimits=np.array([buffer[1] for buffer in buf])
    tag_dict = indexbytag(emoji_list)
    # print(buf)
    # print(emoji_np[0])
    # print(emoji_list[-1])
    s = "language processing"
    polarity, subjectivity = sentiment_predict(s)
    print(polarity)
    print(img_gen(s))
