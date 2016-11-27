import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lg
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score as acc

train_data = pd.read_csv("../datasets/train.csv")

from sklearn.cross_validation import train_test_split
predictors = ['Score', '% Completed']
train_target = train_data['Bootcamp']

x_train, x_test, y_train, y_test = train_test_split(train_data[predictors], train_target) #split training data

# Initialize our algorithm class
alg = RFC()
model = alg.fit(x_train,y_train)

# predictions
predictions = model.predict(x_test)

scores = cross_val_score(model, x_test, y_test, cv=5, scoring= 'f1_weighted')
print scores
print scores.mean()

# accuracy_score
accuracy_score = acc(predictions,y_test)
print accuracy_score

# correlation matrix
correlations = train_data.corr()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,47,1)
ax.set_yticks(ticks)
ax.set_xticks(ticks)
ax.set_xticklabels(train_data.columns, rotation='vertical')
ax.set_yticklabels(train_data.columns)
plt.show()
