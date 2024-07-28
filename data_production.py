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

#%% Load data into Python
# Data from https://www.kaggle.com/datasets/pavankrishnanarne/global-stock-market-2008-present

years_included = range(2008, 2023) # max range 2008 -> 2023
data = pd.read_csv(osjoin("Global_Market_Data", str(years_included[0]) + "_Global_Markets_Data.csv"))
for year in years_included[1:]: # join each years data into one dataframe
    data = pd.concat([data, pd.read_csv(osjoin("Global_Market_Data", str(year) + "_Global_Markets_Data.csv"))], ignore_index=True) 