# Data Acquisition

This section describes the fraud_data.csv dataset and imports it for further processing and analysis of the incidence of fraud in credit card transactions.  This project focuses on selecting the appropriate model evaluation metrics when classes are imbalanced.

Data preparation for the analysis is described in the [next section](https://eagronin.github.io/credit-card-fraud-prepare/).

This project is based on assignments from Applied Machine Learning in Python by University of Michigan on Coursera.

The dataset fraud_data.csv was downloaded from the Coursera website.  Each row in fraud_data.csv corresponds to a credit card transaction. Features include confidential variables V1 through V28 as well as Amount which is the amount of the transaction.

The target is stored in the "class" column, which takes the value of 1 for fraudulent transactions and 0 otherwise.

The following function imports the data:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pylab

# Import the data from fraud_data.csv. 
def read_transactions_data():
    
    df = pd.read_csv('/Users/eagronin/Documents/Data Science/Portfolio/Project Data/Credit card fraud data.csv')
    
    return df
```

Next step: [Data Preparation](https://eagronin.github.io/credit-card-fraud-prepare/)
