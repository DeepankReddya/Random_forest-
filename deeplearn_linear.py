# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
x = np.array([[4],[5],[6],[7],[8],[9],[10]])
y = np.array([[8, 10 , 12, 14, 16, 18, 20]])
# Define model
model = Sequential()
model.add(Dense(1000, input_dim=1, activation= "relu"))
model.add(Dense(1000, activation= "relu"))
model.add(Dense(50, activation= "relu"))
model.add(Dense(1))
#model.summary() #Print model Summarymodel.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(x,y,epochs=200)
x=np.array([[12.0]])
print(model.predict(x))
