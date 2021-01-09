import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import tensorflow as tf 
value_dict={'positive':1, 'negative':0}

# 一些常量
embedding_vector_size = 200
maxlen = 200
test_size = 0.2
# tf.compat.v1.disable_eager_execution()
"""
 读取训练集并构造训练样本
"""
rex = re.compile(r'[!"#$%&\()*+,./:;<=>?@\\^_{|}~]+')
lemmatizer = WordNetLemmatizer()

def clean_review(raw_review: str) -> str:
    # 1. 评论是爬虫抓取的，存在一些 html 标签，需要去掉
    review_text = raw_review.replace("<br />", '')
    # 2. 标点符号只保留 “-” 和 上单引号
    review_text = rex.sub(' ', review_text)
    # 3. 全部变成小写
    review_text = review_text.lower()
    # 4. 分词
    word_list = review_text.split()
    # 5. 词性还原
    tokens = list(map(lemmatizer.lemmatize, word_list))
    lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))
    # # 6. 去停用词
    # meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))
    return lemmatized_tokens



# #求句子长度中位数，确定seq长度=200
# cal_len = pd.DataFrame()
# cal_len['review_lenght'] = list(map(len, sentences))
# me =  cal_len['review_lenght'].median()
# print("中位数：", me)
# print("均值数：", cal_len['review_lenght'].mean())
# from matplotlib import pyplot as plt 
# fig, ax = plt.subplots()
# ax.hist(cal_len['review_lenght'], bins = 200)
# ax.axvline(me,color='orange', label='中位数')
# # plt.axhline(cal_len['review_lenght'].median())
# plt.title('review length distribution')
# plt.show()

# plt.savefig('review_length_distribution.png')
# del cal_len
# # exit
# input('type to continue')
if __name__ =='__main__':
    data = pd.read_csv(r"IMDB Dataset.csv", sep=',')
    # data = data[:100]
    sentences  = data.review.apply(clean_review)
    """
    训练Word2Vec
    """
    # 嵌入的维度
    w2v_model = Word2Vec(
        sentences=sentences,
        size=embedding_vector_size,
        min_count=2, window=3, workers=4)

    w2v_model.save('w2v.model')

    # 取得所有单词
    vocab_list = list(w2v_model.wv.vocab.keys())
    # 每个词语对应的索引
    word_index = {word: index for index, word in enumerate(vocab_list)}
    X_data = list(map(get_index, sentences))
    # 截长补短
    X_pad = pad_sequences(X_data, maxlen=maxlen)
    # 取得标签
    Y = data.sentiment.values
    Y = [value_dict[y] for y in Y]
    Y = np.array(Y)

    np.save('X_pad', X_pad)
    np.save('Y', Y)

# 序列化
def get_index(sentence, word_index):
    # global word_index
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence



# # 划分数据集
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_pad,
#     Y,
#     test_size=test_size,
#     random_state=42)

# """
#  构建分类模型
# """
# # 让 Keras 的 Embedding 层使用训练好的Word2Vec权重
# embedding_matrix = w2v_model.wv.vectors

# model = Sequential()
# model.add(Embedding(
#     input_dim=embedding_matrix.shape[0],
#     output_dim=embedding_matrix.shape[1],
#     input_length=maxlen,
#     weights=[embedding_matrix],
#     trainable=False))
# model.add(Flatten())
# model.add(Dense(5))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(
#     loss="binary_crossentropy",
#     optimizer='adam',
#     metrics=['accuracy'])

# history = model.fit(
#     x=X_train,
#     y=Y_train,
#     validation_data=(X_test, Y_test),
#     batch_size=4,
#     epochs=10)