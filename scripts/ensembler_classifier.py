import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score as acc
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv("./datasets/train.csv")

from sklearn.cross_validation import train_test_split
predictors = ['Score', '% Completed']
train_target = train_data['Bootcamp']

abundant_class_data = train_data[train_data['Bootcamp'] == False]
minority_class_data = train_data[train_data['Bootcamp'] == True]
print abundant_class_data.shape

def generateRandomSamples(df):
    samples = []
    data = df
    for i in xrange(55):
        df_sample = data.sample(n=125)
        samples.append(df_sample.append(minority_class_data))
        data = data.drop(df_sample['S/N'].tolist(), axis=0, errors='ignore')
    return samples
random_samples = generateRandomSamples(abundant_class_data)

classifiers = []
weights = []
for i in xrange(55):
    classifiers.append(LogisticRegression(random_state=0))
    weights.append(1)

eclf = EnsembleVoteClassifier(clfs=classifiers, weights=weights, voting='soft')

train_sample_x = random_samples[1][predictors]
train_sample_target = random_samples[1]['Bootcamp']