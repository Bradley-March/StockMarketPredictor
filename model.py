# Description: 
# Created: July 2024
# Author: Bradley March

#%% Python Preamble

# import necessary packages/functions
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data_production import get_dataset
from time import time

# set up interactive figure windows
mpl.use("TkAgg")
plt.ion()

tstart = time() # start time for the script


#%% Load data from the data_production.py script

number_of_lag_days = 5

data = get_dataset(years_included=range(2008, 2023+1), market="^FTSE", number_of_lag_days=number_of_lag_days)

#%% Split the data into training and testing sets

# define the features and target variable
target = "Daily Return"
features_names = ["Daily Return", "Spread", "Volume"]
# add lag features to the features list
features = []
for col in features_names:
    for i in range(1, number_of_lag_days+1):
        features += [(f"{col}_lag_{i}")]

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
print("Mean Squared Error: {:.4f}".format(mse))

# get the actual values for the last n predictions
y_actual = np.array(y)[-y_pred.size:]

#%% Plot the actual vs predicted values

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(14, 6))
ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
ax1.plot(y_actual, 'b-', alpha=0.3)
ax1.plot(y_pred, 'r-', alpha=0.3)
ax2.plot(y_actual -  y_pred, 'k.-')
fig.tight_layout()

#%% Timings
tend = time() # end time for the script
print("Time taken: {:.2f} seconds".format(tend - tstart))