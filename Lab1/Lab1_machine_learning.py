import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Weather data
# ------------

# Goal: Given a description of the weather today, decide whether to go play outdoor
# (e.g., tennis) or not.

# Data Set Characteristics:  Multivariate, supervised
# Number of Instances (observations): 14
# Attribute (features) Characteristics: Categorical
# Number of Attributes: 4
# Associated Task: Classification
# Number of classes: 2

# Data file description:

# - The first line is the column names
# - Columns 1-4 are the input features
# - Column 5 is the target class

# Attributes:

# 1) Outlook (overcast, rainy, sunny)
# 2) Temperature (hot, cool, mild)
# 3) Humidity (high, normal)
# 4) Windy (FALSE, TRUE)

# Classes:

# Play (yes, no)


# Load the data
def load_data(filename):
    data = pd.read_csv(filename)
    return data

def classification(training_set, test_set):
    n=len(training_set.columns)
    dp=len(training_set.lines)
    Outlook ={'overcast':0, 'rainy':0, 'sunny':0}
    Temperature = {'hot':0, 'cool':0, 'mild':0}
    Humidity = {'high':0, 'normal':0}
    Windy = {'False':0, 'True':0}
    for i in range(len(training_set.columns)):
        for j in range(len(training_set.lines)):



data = load_data('dataset.csv')
print(data)
training_set = data.head(10)
print(training_set)
test_set = data.tail(4)
print(test_set)
#test_set = test_set.iloc[:, :-1]
#print(test_set)

