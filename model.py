# Description: 
# Created: July 2024
# Author: Bradley March

#%% Python Preamble

# import necessary packages/functions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data_production import get_dataset
from time import time

tstart = time() # start time for the script


#%% Load data from the data_production.py script

data = get_dataset(years_included=range(2008, 2023+1), market="^FTSE")

#%% Split the data into training and testing sets

# define the features and target variable
features = ["Daily Return", "Spread", "Volume"]
target = "Daily Return"

# split the data into features and target variable
X = data[features]
y = data[target]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

#%% Train a linear regression model

# initialise the model
model = LinearRegression()

# fit the model
model.fit(X_train, y_train)

#%% Evaluate the model

# make predictions
y_pred = model.predict(X_test)

# calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# get the actual values for the last n predictions
y_actual = np.array(y)[-y_pred.size:]


#%% Timings
tend = time() # end time for the script
print("Time taken: {:.2f} seconds".format(tend - tstart))