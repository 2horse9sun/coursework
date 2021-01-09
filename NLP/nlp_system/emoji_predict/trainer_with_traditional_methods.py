import numpy as np
# 载入数据
X_pad = np.load('../data/X_pad.npy')
Y = np.load('../data/Y.npy')
print('data loaded..')
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, Y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
from sklearn import naive_bayes
# from sklearn.externals 
import joblib
from sklearn.metrics import accuracy_score,classification_report
clf1 = SVC()
print('tring to fit in clf1..')
clf1.fit(X_train,Y_train)
print('svm model generated..')
joblib.dump(clf1, '../models/svm.pkl')
y1_pred = clf1.predict(X_test)
print(classification_report(Y_test, y1_pred))
print('tring to fit in clf1..')
clf2 = naive_bayes.MultinomialNB(alpha=0.5)
clf2.fit(X_train,Y_train)
print('bys model generated..')
y2_pred = clf2.predict(X_test)
joblib.dump(clf2, '../models/bys.pkl')
print(classification_report(Y_test, y2_pred))

