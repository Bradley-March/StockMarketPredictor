#%% Python Preamble

# import necessary packages/functions
import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join as osjoin
from time import time

# set up interactive figure windows
mpl.use("TkAgg")
plt.ion()

tstart = time() # start time for the script

#%% Load data into Python
# Data from https://www.kaggle.com/datasets/pavankrishnanarne/global-stock-market-2008-present

years_included = range(2008, 2023) # max range 2008 -> 2023
data = pd.read_csv(osjoin("Global_Market_Data", str(years_included[0]) + "_Global_Markets_Data.csv"))
for year in years_included[1:]: # combine every year into one dataframe
    data = pd.concat([data, pd.read_csv(osjoin("Global_Market_Data", str(year) + "_Global_Markets_Data.csv"))], ignore_index=True) 

print(data.groupby("Ticker").size().sort_values(ascending=False)) # print number of entries for each ticker 

#%% Create a subset of the data for a single ticker

# select a single ticker
market = "^FTSE"
market_data = data[data["Ticker"] == market]

#%% Modifying the data

# drop irrelevant columns
# (Open same as porevious day's close, Adj Close same as Close)
market_data = market_data.drop(columns=["Open", "Adj Close"])

# convert date to datetime
data['Date'] = pd.to_datetime(data['Date'])
# sort by date
data = data.sort_values("Date").reset_index(drop=True)

#%% Convert daily price to daily percentage change

# calculate daily percentage change
market_data["Daily Return"] = market_data["Close"].pct_change()

# drop the first row as it will be NaN
market_data = market_data.dropna()

#%% Add a columns for the spread between the high and low price

# calculate the spread
market_data["Spread"] = (market_data["High"] - market_data["Low"]) / market_data["Close"]

#%% Plot the data

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))
# plot the daily return
ax1.plot(market_data["Date"], market_data["Daily Return"], 'k.')
ax1.set_ylabel("Daily Percent Change")
# plot the daily spread
ax2.plot(market_data["Date"], market_data["Spread"], 'r.')
ax2.set_xticks(market_data["Date"][::250])  # set x-ticks to roughly yearly intervals
ax2.set_ylabel("Spread [(High - Low) / Close]")

fig.autofmt_xdate() # rotate x-axis labels
fig.tight_layout() # adjust subplots to fit into figure area

#%% Final timings 
tend = time() # end time for the script
print(f"Time taken: {tend - tstart:.2f} seconds") # print time taken for the script to run