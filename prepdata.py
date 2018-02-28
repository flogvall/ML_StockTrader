# Erik Flogvall
# Udacity Machine Learning Capstone Project
# prepdata.py contains function for importing and preparing data before usage in the Q-learner

import os
import numpy as np
import pandas as pd
import random

def get_data(dir):
    '''
    get_data reads the csv data files for all stock in a given directory.
    The outputs are a Pandas dataframe with the closing price at every date and
    stock, and a dictionary with lists containing the first and last trading
    days for each stock.
    '''
    # The file names for all csv files (one for each stock).
    file_names = os.listdir(dir)

    # Initializes an empty dictionary for storing each stocks first and last trading days.
    trading_days = {}

    # Loops through all files and stores the closing prices in teh all_data Pandas dataframe and the trading days in trading_days.
    for n,file in enumerate(file_names):

        # Gets the stock indicator (name) from the file name
        name = file.replace("_data.csv","")

        # Reads the csv and stores the dates and closing price into a Pandas dataframe
        temp_data = pd.read_csv(dir + "/" + file, sep=",", usecols = ["Date","Close"])

        # Converts the date to datetime format
        temp_data["Date"] = pd.to_datetime(temp_data["Date"])

        # Set the date as the index for the stocks dataframe
        temp_data = temp_data.set_index("Date")

        # Checks if there is any data in the Pandas dataframe
        if not temp_data.empty:

            # Gets the first and last trading days for the current stock
            trading_days[name] = [temp_data.index[0],temp_data.index[-1]]

            # Renames the closing price column in the dataframe to the stock name
            temp_data.rename(columns={'Close': name}, inplace=True)

            # If this is first stock that is read it will initialize a all_data pandas dataframe where all the closing prices will be stored
            if n == 0:
                all_data = temp_data
            # Combines the current stocks dataframe with the all_data dataframe
            else:
                # Adds the current stock into all_data as a new column
                all_data = pd.concat([all_data, temp_data], axis=1)

    # Returns a dataframe with closing prices for all stocks and a dictionary
    return all_data, trading_days


def prepare_data(dir, split_ratio, window_size):
    '''Preprocesses the data to create new features from the closing prices of
    the stocks and splits the data into a training only set and a training
    & evaluation set.

    The training only set is used to simulate creating
    Q-learner from historical data and the training & evaluation set is used
    to simluate continute training the Q-learner on new data while evaluating
    the performance of the Q-learner.

    The inputs for this functions is dir - the location of the folder containing
    the data files, split_ratio - the ratio of the dataset used for the training
    only set and window_size - the length of the window used for creating
    rolling statistics'''

    # Gets the read closing prices and trading days from the get_data_function
    all_data, trading_days = get_data(dir)

    # Initializes the prepared_data dict containing a dict for training only
    # called 'train_only' and another dict for training & evaluation called
    # 'train_eval'
    prepared_data = {'train_only':{},'train_eval':{}}

    # Gets an list with all dates in the dataset
    dates = all_data.index

    # Computes at what index of the dates list that the dataset should be split
    # into a training only set and a training & evaluation set.
    split_num = int((len(dates)-window_size)*split_ratio+window_size)

    # Gets the split date with the computed index
    split_date = dates[split_num]

    # Splits the dates into 'train_only' and 'train_eval'
    dates = {'train_only':dates[(window_size-1):split_num],'train_eval':dates[split_num:]}

    # Gets an list with all the stock names (indicators)
    stocks = all_data.columns

    # Loops through all stocks and sorts the stocks into subdictionaries of
    # prepared_data
    for stock in stocks:

        # Gets the first trading day for the stock
        start_date = trading_days[stock][0]

        # Gets the last trading day for the stock
        end_date = trading_days[stock][1]

        # Makes a temporary Pandas dictonary for the current stock between the
        # first and last trading days
        temp_data = all_data[stock].loc[start_date:end_date]

        # Uses the forward fill method to fill any trading days don't have any
        # closing price for the current stock
        temp_data = temp_data.fillna(method = "ffill")

        # Renames the stock name to 'Price' as to have an uniform when
        # the data has been sorted into seperate dictionaries
        temp_data.rename(columns={stock: 'Price'}, inplace=True)

        # Uses the add_finiancial_stats function to add new financial features
        # computed from the closing price
        temp_data = add_finiancial_stats(temp_data, window_size)

        # Updates the start date as using rolling statistics means that the
        # the first date with the new features will be put forward depending on the
        # rolling window size
        start_date = temp_data.index[0]

        # Updates the trading_days dictionary
        trading_days[stock] = [start_date, end_date]

        # Splits the data into a training only set and a training & evaluation set
        if split_date > start_date and split_date < end_date:
            prepared_data['train_only'][stock] = temp_data.loc[:split_date]
            prepared_data['train_only'][stock] = prepared_data['train_only'][stock].iloc[:-1]
            prepared_data['train_eval'][stock] = temp_data.loc[split_date:]
        elif split_date <= start_date:
            prepared_data['train_eval'][stock] = temp_data
        elif split_date >= end_date:
            prepared_data['train_only'][stock] = temp_data

    # Output the updated data and lists of all dates and stocks
    return prepared_data, trading_days, dates, stocks



def add_finiancial_stats(price, window_size):
    '''Adds new feautres to data computed from the closing price. The 'price'
    input is a Pandas dataframe containg date and the closing price'''

    # Computes the rolling mean of the closing price
    rolling_mean = price.rolling(window_size).mean()
    # Normalizes the rolling mean with the closing price
    rolling_mean = (rolling_mean - price) / price

    # Computes the rolling standard deviation of the closing price
    rolling_std = price.rolling(window_size).std()
    # Normalizes the rolling mean with the closing price
    rolling_std = (rolling_std - price) / price

    # Computes the daily return
    daily_return = ((price / price.shift(1)) -1)*100

    # Computes the daily return for 1 day ago
    daily_return_m1 = daily_return.shift(1)

    # Computes the daily return for 2 days ago
    daily_return_m2 = daily_return.shift(2)

    # Computes the daily return for 3 days ago
    daily_return_m3 = daily_return.shift(3)

    # Computes the upper and lower bollinger bands
    upper_bollinger = rolling_mean + 2*rolling_std
    lower_bollinger = rolling_mean - 2*rolling_std
    # Normalizes the bollinger bands
    upper_bollinger = (upper_bollinger - price) / price
    lower_bolliger = (lower_bollinger - price) / price

    # Makes a dictionary used for creting a new dataframe with the new feautres included
    dfdict = {'Price': price, 'Daily_return': daily_return, 'Rolling_mean': rolling_mean, 'Rolling_std': rolling_std, 'Upper_bollinger': upper_bollinger, 'Lower_bollinger': lower_bolliger, 'Daily_return_-1': daily_return_m1, 'Daily_return_-2': daily_return_m2}

    # Creates the new dataframe
    data = pd.DataFrame(dfdict)

    # Removes the first entries that don't have any values as because of the rolling stats
    data = data.iloc[(window_size-1):]
    return data
