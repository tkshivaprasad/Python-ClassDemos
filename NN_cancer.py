import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets



from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer.keys()

# Print full description by running:
# print(cancer['DESCR'])
# 569 data points with 30 features
cancer['data'].shape

#Let's set up our Data and our Labels

X = cancer['data']
y = cancer['target']


#split our data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

#scale your data
#must apply the same scaling to the test set for meaningful results
#use the built-in StandardScaler for standardization.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Multi-Layer Perceptron Classifier model
from sklearn.neural_network import MLPClassifier

#there are a lot of parameters you can choose to define and customize here
#we will only define the hidden_layer_sizes
#this parameter you pass in a tuple consisting of the number of neurons you want at each layer,
#where the nth entry in the tuple represents the number of neurons in the nth layer of the MLP model

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))

#we can fit the training data to our model,
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

#classification report and confusion matrix to evaluate how well our model performed
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))



