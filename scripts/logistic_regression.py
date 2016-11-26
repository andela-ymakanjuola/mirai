import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lg
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score as acc

train_data = pd.read_csv("./datasets/train.csv")

from sklearn.cross_validation import train_test_split
predictors = ['Score', '% Completed']
train_target = train_data['Bootcamp']

abundant_class_data = train_data[train_data['Bootcamp'] == False]
minority_class_data = train_data[train_data['Bootcamp'] == True]
print abundant_class_data.shape

samples = []
def random_sampling(df):
    data = df
    for i in xrange(55):
        df_sample = data.sample(n=125)
        samples.append(df_sample.append(minority_class_data))
        data = data.drop(df_sample['S/N'].tolist(), axis=0, errors='ignore')
random_sampling(abundant_class_data)

train_sample_x = samples[1][predictors]
train_sample_target = samples[1]['Bootcamp']
x_train, x_test, y_train, y_test = train_test_split(train_sample_x, train_sample_target) #split training data

# The columns we'll use to predict the target
# train_predictors = train_data[predictors]
# print train_predictors
print x_train['Score'].describe()
# print np.log(x_train['Score'])

# normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(x_train)

# Initialize our algorithm class
alg = lg()
model = alg.fit(x_train,y_train)

# predictions
predictions = model.predict(x_test)

scores = cross_val_score(model, x_test, y_test, cv=5)
print scores

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
