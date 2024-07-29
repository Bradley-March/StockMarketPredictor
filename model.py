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

number_of_lag_days = 1

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

#%% Calculate when the model agrees with the direction the stock moves

correct = np.sign(y_actual) == np.sign(y_pred)
# print percent of times model agrees with actual
print("Percent of times model agrees with actual", 100 * np.sum(correct) / correct.size)

# print percent of times the actual value is positive
print("Percent of times the actual value is positive", 100 * np.sum(y_actual > 0) / y_actual.size)

#%% Plot the actual vs predicted values

xrange = np.arange(len(y_actual))
dates = pd.to_datetime(data["Date"].values[-y_pred.size:])

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(14, 6))
ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
ax1.plot(dates, y_actual, 'k-', alpha=0.3)
ax1.plot(dates, y_pred, 'b-', alpha=0.3)
ax1.plot(dates[correct], y_actual[correct], 'g.', markersize=2)
ax1.plot(dates[~correct], y_actual[~correct], 'r.', markersize=2)
ax2.plot(dates, y_actual -  y_pred, 'k.-')
# formatting
ax2.set_xticks(dates[::20]) # set x-ticks to roughly monthly intervals
fig.autofmt_xdate() # rotate x-axis labels
fig.tight_layout() # adjust subplots to fit into figure area

#%% Profit and loss calculation

# get opening/closing values
y_open = data["Close"].values[-1-y_pred.size:-1]
y_close = data["Close"].values[-y_pred.size:]

# calculate the profit and loss
pnl = np.sum((y_close - y_open)[y_pred > 0])
pnl_holding = y_close[-1] - y_open[0]

# calculate the percentage pnl
pnl_percent = 100 * pnl / y_open[0]
pnl_holding_percent = 100 * pnl_holding / y_open[0]

# calculate the amount of time between start and end
time_diff = dates[-1] - dates[0]
# convert to years
time_diff = time_diff.days / 365.25

print("Profit and Loss: £{:.2f}, {:.2f}% increase, {:.2f}% annual increase".format(pnl, pnl_percent, pnl_percent / time_diff))
print("Holding profit and loss: £{:.2f}, {:.2f}% increase, {:.2f}% annual increase".format(pnl_holding, pnl_holding_percent, pnl_holding_percent / time_diff))



#%% Timings
tend = time() # end time for the script
print("Time taken: {:.2f} seconds".format(tend - tstart))