from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score as acc,f1_score,roc_auc_score

train_data = pd.read_csv("../datasets/train.csv")
test = pd.read_csv("../datasets/test.csv")
#train_data['score_v_completed']=train_data['Score']//train_data['% Completed']
static_fields = []
for c in train_data.columns:
    if len(list(train_data[c].unique()))<=1:
        static_fields.append(c)

static_fields
train_data=train_data.drop(static_fields,axis=1)
X=train_data.drop(['Joined Proctor on','Stage','% Score','Entry','Bootcamp'],axis=1)
test_data = test.drop(static_fields,axis=1)
test_data = test_data.drop(['Joined Proctor on','% Score', 'Bootcamp'],axis=1)
y=train_data['Bootcamp']
#x_train, x_test, y_train, y_test = train_test_split(X, y) #split training data

# x_train=x_train[x_train['Score']>=92]
x_train=X.drop(['S/N'],axis=1)
# leak_test=x_test[x_test['Score']>=92]
# leak_test=list(leak_test['S/N'])
index=test_data['S/N']
x_test=test_data.drop(['S/N'],axis=1)
clf = LogisticRegression()
grad = GradientBoostingClassifier(n_estimators=2000, max_depth=12, random_state=42)
print 'x_train', x_train.info()
print 'x_test',x_test.info()
grad.fit(x_train,y)

enc = OneHotEncoder()
enc.fit(grad.apply(x_train)[:, :, 0])
decomp = TruncatedSVD(n_components=3, random_state=42)
decomp.fit(x_train,y)

x_train = enc.transform(grad.apply(x_train)[:, :, 0])
x_test = enc.transform(grad.apply(x_test)[:, :, 0])
clf.fit(x_train,y)

scores = cross_val_score(clf, x_train, y, cv=5, scoring='f1_weighted')
print 'F1 score', scores

predictions=clf.predict(x_test)

accuracy_score = acc(test['Bootcamp'],predictions)
print 'accuracy',accuracy_score

accuracy_score = f1_score(test['Bootcamp'],predictions,average='weighted')
print 'f1 on test', accuracy_score

accuracy_score = roc_auc_score(test['Bootcamp'],predictions)
print 'ROC', accuracy_score
