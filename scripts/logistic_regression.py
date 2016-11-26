import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lg
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score as acc

train_data = pd.read_csv("../datasets/train.csv")

print train_data['Entry'].unique()

# The columns we'll use to predict the target
predictors = []

# Initialize our algorithm class
# alg = lg()
# model = alg.fit()

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
