import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as lg
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score as acc

train_data = pd.read_csv("../datasets/train.csv")
print(train_data.head(5))
print(train_data.describe())

# print train_data['Stage'].unique()

# The columns we'll use to predict the target
predictors = []

# Initialize our algorithm class
# alg = lg()
# model = alg.fit()
