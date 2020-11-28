from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np 
analyzer = SentimentIntensityAnalyzer()
subjective_limit = 0.1
api_use=True
def load_data(path):
    xvecs = np.load(path+'tfvecMatrix.npy')
    ylabels = np.load(path+'yLabel.npy')
    return xvecs, ylabels
    
def sentiment_predict(sentence):
    tblob = TextBlob(sentence)
    vs = analyzer.polarity_scores(sentence)
    polarity = (tblob.polarity+vs['pos']-vs['neg'])/2
    return polarity, tblob.subjectivity

def img_gen(sentence):
    po_score, sub_score = sentiment_predict(sentence)
    if sub_score>subjective_limit:
        if po_score>0:
            uni = "\u0001f60a"
        else:
            uni = "\u0001f622"
    else:
        pass
        #TODO: generate accoring to keywords
    return uni

s = "I really like this fine wheather"
polarity, subjectivity = sentiment_predict(s)
# if polarity>0:

print(sentiment_predict(s))
print(img_gen(s))