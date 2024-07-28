# Description: This script loads data from the Global_Market_Data folder and creates a subset of the data for a single ticker.
#             It then modifies the data by calculating the daily percentage change and the spread between the high and low price.
# Created: July 2024
# Author: Bradley March
# Data from https://www.kaggle.com/datasets/pavankrishnanarne/global-stock-market-2008-present

#%% Python Preamble

# import necessary packages/functions
import numpy as np
import pandas as pd
from os.path import join as osjoin

#%% Set up functions to produce the data set

def get_dataset(years_included=range(2008, 2023+1), market="^FTSE"):
    """
    Input:  years_included - list of years to include in the dataset (min 2008, max 2023)
            market - the ticker for the market to include in the dataset
    Output: market_data - a pandas dataframe containing the data for the specified market
    Load the data from the Global_Market_Data folder and return a subset of the data for a single ticker.
    """
    # load data into Python
    data = pd.read_csv(osjoin("Global_Market_Data", str(years_included[0]) + "_Global_Markets_Data.csv"))
    for year in years_included[1:]: # combine every year into one dataframe
        data = pd.concat([data, pd.read_csv(osjoin("Global_Market_Data", str(year) + "_Global_Markets_Data.csv"))], ignore_index=True) 
    
    # create a subset of the data for a single ticker
    market_data = data[data["Ticker"] == market]
    # drop irrelevant columns
    # (Open same as porevious day's close, Adj Close same as Close)
    market_data = market_data.drop(columns=["Open", "Adj Close"])
    # convert date to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    # sort by date
    data = data.sort_values("Date").reset_index(drop=True)

    # calculate daily percentage change
    market_data["Daily Return"] = market_data["Close"].pct_change()
    # drop the first row as it will be NaN
    market_data = market_data.dropna()
    # calculate the spread
    market_data["Spread"] = (market_data["High"] - market_data["Low"]) / market_data["Close"]

    return market_data

#%% Set up the script to produce the data set

if __name__ == "__main__": 
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from time import time

    # set up interactive figure windows
    mpl.use("TkAgg")
    plt.ion()

    tstart = time() # start time for the script

    market_data = get_dataset() # get the dataset

    # plot the data
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))
    # plot the daily return
    ax1.plot(market_data["Date"], market_data["Daily Return"], 'k.')
    ax1.set_ylabel("Daily Percent Change")
    # plot the daily spread
    ax2.plot(market_data["Date"], market_data["Spread"], 'r.')
    ax2.set_xticks(market_data["Date"][::250])  # set x-ticks to roughly yearly intervals
    ax2.set_ylabel("Spread [(High - Low) / Close]")
    # formatting
    fig.autofmt_xdate() # rotate x-axis labels
    fig.tight_layout() # adjust subplots to fit into figure area

    tend = time() # end time for the script
    print(f"Time taken: {tend - tstart:.2f} seconds") # print time taken for the script to run