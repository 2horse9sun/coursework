import numpy as np
# 载入数据
X_pad = np.load('../data/X_pad.npy')
Y = np.load('../data/Y.npy')
maxlen = 200
epochs = 20
batch_size = 50
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, Y, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, Flatten
from keras.callbacks import ModelCheckpoint

from gensim.models import Word2Vec
w2v_model = Word2Vec.load('../data/w2v.model')
# 让 Keras 的 Embedding 层使用训练好的Word2Vec权重
embedding_matrix = w2v_model.wv.vectors

model = Sequential()
model.add(Embedding(
    input_dim=embedding_matrix.shape[0],
    output_dim=embedding_matrix.shape[1],
    input_length=maxlen,
    weights=[embedding_matrix],
    trainable=False))
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy'])

filepath = "../cnn_models/cnn_model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, 
                            monitor='val_accuracy', 
                            verbose=0, 
                            save_best_only=True, 
                            save_weights_only=False, 
                            mode='auto', 
                            period=epochs)

history = model.fit(
    x=X_train,
    y=Y_train,
    validation_data=(X_test, Y_test),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[checkpoint]
    )

# 绘图
import matplotlib.pyplot as plt 
print(history.history)
val_accuracy_list = np.asarray(history.history['val_accuracy'])
accuracy_list = np.asarray(history.history['accuracy'])
x = np.arange(epochs)
plt.plot(x,accuracy_list, label='accuracy',color='blue')
plt.plot(x,val_accuracy_list, label='val_accuracy',color='red')
plt.legend()
plt.show()
plt.savefig('cnn accuracy grow by epoch.png')