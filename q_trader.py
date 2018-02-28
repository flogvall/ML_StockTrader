# Erik Flogvall
# Udacity Machine Learning Capstone Project
# q_trader.py contains the class QTrader that defines functions and variables for the Q-learner used for trading stock

import os
import numpy as np
import pandas as pd
import random

class QTrader():
    '''QTrader is a class containing the Q-learner used for making the trading
    agent. '''

    def __init__(self, qt_input, epsilon = 1.0, alpha = 0.5,epsilon_decay = 4e-6, gamma = 0.0):
        '''The init function used for creating a new QTrader agent'''

        # The random exploration factor
        self.epsilon = epsilon

        # The learning rate
        self.alpha = alpha

        # The decay rate for the random exploration factor
        self.epsilon_decay = epsilon_decay

        # The discount rate for future rewards
        self.gamma = gamma

        # The data with the computed features. qt_input is a input dictionary
        # containing data and settings for the problem
        self.data = qt_input['data']

        # A list containing all dates from the dataset
        self.dates = qt_input['dates']

        # A list with all stock names
        self.stocks = qt_input['stocks']

        # Setting used for making the states. Its a dictionary with the features
        # to be used with the number of bins used for discretization as the value
        self.state_settings = qt_input['state_settings']

        # A dictionary with all trading days for all stocks
        self.trading_days = qt_input['trading_days']

        # A string with the type of learning used:
        # 'train_only' - training without any evaluation. Simulates making a
        # Q-learner with historical data
        # 'train_eval' - training with evaluation. Simulates implementing the
        # Q-learner on new data
        self.learning = 'train_only'

        # Initializes a counter used for decay epsilon
        self.t = 0

        # The current day. Initialized as None.
        self.today = None

        # The next day. Initialized as None.
        self.next_day = None

        # Stores the starting epsilon before decaying
        self.epsilon_start = epsilon

        # Initializes a log dict used logging the performance and progress of
        # the Q-learner
        self.log = {'t':[],'epsilon':[],'alpha':[],'reward':[], 'action':[], 'state':[], 'stock':[], 'balance':[], 'train':[]}

        # Computes at what interval the bins used for discretization should be
        # for each feature.
        self.bins =  self.make_bins()

        # Makes states for every day and stock with the previously made bins
        self.states = self.make_states_from_data()

        # Initializes the Q-table
        self.Q = self.init_Q_table()

        # Initializes a T-table (expected transitions) used for Dyna
        self.T = self.init_T_table()

        # Initializes a R-talbe (expected rewards) used for Dyna
        self.R = self.init_R()

        # Initializes a ledger containng the balance (profit) for each stock
        self.stock_ledger = self.init_stock_ledger()

    def make_states_from_data(self):
        ''' Makes states for all data using the bins created for every
        feature selected to be used'''

        # Initializes an empty dictionary for keeping the states made
    	states = {}

        # Loops through the two splitted parts of the data: 'train_only' and 'train_eval'
        for train in self.data:

            # Initializes a subdictionary for keeping the states for each stock
    		subdict = {}

            # Loops through each stock keept in the current train-set
    		for stock in self.data[train]:

                # Gets the continuous data
    			stock_data = self.data[train][stock]

                # Gets the dates for the current stock
    			days = stock_data.index

                # Gets the features used for making states
    			features = self.bins.keys()

                # Initializes a list with a tuple for every day
    			state_array = [()]*len(days)

                # Loops through the features
    			for feature in features:

                    # Gets the values for all dates of the current feature and stock
    				values = stock_data[feature].values

                    # Loops through the values to discretize them and add the to the tuple of its day
    				for n, val in enumerate(values):

                        # Discretizes the value and adds it to the tuple
    					state_array[n] = state_array[n] + (self.discretize(val,feature),)

                # Creates a subdictionary for the stock with the dates and states.
    			subdict[stock] = dict(zip(days,state_array))
    		states[train] = subdict
    	return states

    def make_bins(self):
        ''' make_bins creates bin from the state settings and the data in the 'train_only' training set'''

        # Initializes a empty dictionary for storing arrays of all selected features.
    	features_arrays = {}

        # Loop through all stocks in the 'train_only' set.
    	for stock in self.data['train_only']:
            # Gets the data for the stock in the 'train_only' set.
    		stock_data = self.data['train_only'][stock]

            # Loops through all features selected to make the state
    		for feature in self.state_settings:

                # Checks if the feature has been added to dictionary
    			if feature in features_arrays:
                    # Appends the arrary of the feature with the values of the current stock and feature
    				features_arrays[feature] = np.append(features_arrays[feature],stock_data[feature].values)
    			else:
                    # If the feature is not yet added to the dictionary it is initialized with the values of the current stock and feature
    				features_arrays[feature] = stock_data[feature].values

        # Initializes an empty dictionary for keeping the bin limits in.
        bins = {}

        # Loops through the features added to feature array
    	for feature in features_arrays:
            # Initializes a list for the bin limits of the current feature
    		bins[feature] =  []

            # Gets the number of bins (discrete steps) for the current feature
    		no_bins = self.state_settings[feature]

            # Computes the width of the bins from the length of feature array
    		bin_width = len(features_arrays[feature])/no_bins

            # Sorts the feature array of the stock
    		sorted_array = np.sort(features_arrays[feature])

            # Creates the bins for the feature
    		for k in range(0,no_bins-1):
                # Gets a which value each bin will end and stores in bins list
    			bins[feature].append(sorted_array[(k+1)*bin_width])
    	return bins

    def discretize(self, value, feature):
        ''' Descritizes a value depending on the feature using the computed bins'''
    	output = None
        # Loops through the bin limits for the feature to select what bin
        # the value should be placed in
    	for n, b in enumerate(self.bins[feature]):
            # If the value is less or equal to bin limit it will output the bin
            # number and break the loop
    		if value <= b:
    			output = n
    			break
        # If the value is larger than highest bin limit it outputs the final bin
    	if value > self.bins[feature][-1]:
    		output = len(self.bins[feature])
    	return output

    def init_Q_table(self):
        '''Initializes a Q-table using the states combined with the ownership state.'''
        # Initializes a empty dict for storing the Q-table
        Q = {}

        # Initializes a empty list for storing possible states for the stocks
    	state_array = []

        # An array used storing the number of discrete step for every selected feature
    	num_array = []

        # Loops through the selected features
    	for feature in self.state_settings:
            # Adds the number of step fore the feature to num_array
    		num_array.append(self.state_settings[feature])

        # Starts a recursive function for looping through possible states
    	state_array = self.recursive_init_Q((),num_array,state_array)

        # Loops through the states generated by the recursive function to add
        # actions to the state
    	for state in state_array:
            # If the state means the stock is currently owned are the possible
            # actions  either to sell or do nothing.
            if state[-1] == 'Owned':
                Q[state] = {'Sell': 0.0, "Nothing": 0.0}

            # If the state means the stock is not owned are the possible
            # actions either to buy or do nothing.
            else:
                Q[state] = {'Buy': 0.0, "Nothing": 0.0}
    	return Q

    def recursive_init_Q(self,input_state,N,state_array):
        ''' Recursive function used for looping through all possible states'''
    	if N != []:
    		for n in range(0,N[0],1):
    			state_array = self.recursive_init_Q(input_state + (n,), N[1:],state_array)
    	else:
    		state_array.append(input_state + ('Owned',))
    		state_array.append(input_state + ('Not owned',))
        return state_array

    def init_T_table(self):
        ''' Function to initialize the T-table that stores the number of state transitions for Dyna'''
        # Initializesa an empty dictionary to log the transitions in.
        T = {}

        # Loops through the states in Q-table as these will be the same states.
        for state in self.Q:
            # What next state is possible will depend on the ownership and action
            if state[-1] == 'Owned':
                # The possble action in if the stock is owened
                T[state] = {'Sell': {}, "Nothing": {}}
                # Loops through the states again to initialize the next states
                for next_state in self.Q:
                    if next_state[-1] == 'Owned':
                        # Initializes the log for the next state with zero if
                        # the action was nothing
                        T[state]['Nothing'][next_state] = 0
                    else:
                        # Initializes the log for the next state with zero if
                        # the action was sell
                        T[state]['Sell'][next_state] = 0
            else:
                # Does the same thing but the actions are buy and nothing when
                # the stock is not owned.
                T[state] = {'Buy': {}, "Nothing": {}}
                for next_state in self.Q:
                    if next_state[-1] == 'Owned':
                        T[state]['Buy'][next_state] = 0
                    else:
                        T[state]['Nothing'][next_state] = 0
    	return T

    def increment_T(self,stock,state,action):
        ''' Increments the value in T-table for the transitions
        that occured with one'''
        next_state = self.get_state(stock,self.next_day)
        self.T[state][action][next_state] += 1

    def init_R(self):
        ''' Initializes the R-table for Dyna that contains the expected reward
        of an action'''
        R = {}
        for state in self.Q:
            R[state] = {}
            for action in self.Q[state]:
                R[state][action] = 0
        return R

    def init_stock_ledger(self):
        ''' Initializes the stock ledger that keeps the balance (profit) for
        stock'''
        # The stock ledger is as a dictionary with the stock names as keys.
    	stock_ledger = {}
    	for stock in self.stocks:
            # Every stock is initialized as not owned with a balance of zero.
    		stock_ledger[stock] = {'ownership': 'Not owned', 'balance': 0}
    	return stock_ledger

    def select_action(self, state, searching_for_argmax,run_dyna):
        ''' Function for selecting the best action for the current state.
        The main input is the current state. The two other inputs are options
        used for altering the usage of the function.

        Use cases:
        1: During Q-learning to find the best action for the current state
           Settings: searching_for_argmax = False, run_dyna = False
        2: During Q-learning to find the best action for the next state when
           discounting future rewards. No random selection.
            Settings: searching_for_argmax = True,
           run_dyna = False
        3: When running Dyna simulations. No random selection and the Q-table is
           updated for every Dyna run and does not use the Daily Q-table.
           Settings: searching_for_argmax = True, run_dyna = True
        '''

        # If run_dyna is set to False will the Q-table for the current day be
        # loaded. This means that the updates from learning will first be
        # availible the next day.
        if not run_dyna:
            # The Q-values for the current state
            q_state = self.Q_today[state]
        # If rund_dyna is set to True will the Q-table with the lastest updates
        # be used.
        else:
            # The Q-values for the current state
            q_state = self.Q[state]

        # If searching_for_argmax is set to False will the random exploration be
        # used for learning
        if not searching_for_argmax:
            # Gets a random number p
            p = random.random()
            # If the random number p is smaller than the random exploration factor
            # and the training mode is train_only will a random action be selected
            if self.learning == 'train_only' and p < self.epsilon:
                for action in q_state:
                    # The Q-values for the current state is temporarily overwritten
                    # with random values.
                    q_state[action] = random.random()

        # Initializes the best action for the state (best_action) and
        # the value (best_value) as None
        best_action = None
    	best_value = None

        # Loops through the actions and evaluates what action is best.
    	for action in q_state:
            # Gets the value of the current action
    		value = q_state[action]

            # If no action has been evaluted is the current action the best.
    		if best_value == None:
    			best_action = action
    			best_value = value

            # If the current action has a higher value than the best_value will
            # the current action be set as the best one.
    		elif best_value < value:
    			best_action = action
    			best_value = value

            # If the best values equals the value for the current action will
            # the current action be used for the best action with a 50% probabilty.
    		elif best_value == value:
    			p3 = random.random()
    			if p3 >= 0.5:
    				best_action = action
    				best_value = value

    	return best_action

    def update_ledger(self, stock, action):
        '''Updates the stock ledger to execute the action.'''

        # Gets the price for the current day (the day after the trade was decided)
        price = self.data[self.learning][stock]['Price'][self.today]

        # If the action is to sell a owned stock:
    	if action == 'Sell' and self.stock_ledger[stock]['ownership'] == 'Owned':
            # The ownership status is changed to 'Not owned'
    		self.stock_ledger[stock]['ownership'] = 'Not owned'
            # The balance of the stock is increased with the current price
    		self.stock_ledger[stock]['balance'] += price

        # If the action is to buy the stock
    	elif action == 'Buy' and self.stock_ledger[stock]['ownership'] == 'Not owned':
            # The ownership status is changed to 'Owned'
    		self.stock_ledger[stock]['ownership'] = 'Owned'
            # The balance of the stock is decreased with the current price
    		self.stock_ledger[stock]['balance'] -= price

    def learn(self, state, action, reward, stock):
        ''' Function for Q-learning with the update rule. Future rewards is discounted'''

        # Gets the next state for the stock
        next_state = self.get_state(stock,self.next_day)

        # Selects the best action for the next state without any randomness with
        # the use of the Q-table for the current day.
        best_action = self.select_action(next_state, True, False)

        # Gets the maximum Q value for the next state
        max_Q = self.Q[next_state][best_action]

        # Updates the Q values for the current state and action with the
        # update rule with discounted future rewards.
        self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*(reward +  self.gamma*max_Q)

    def learn_dyna(self, state, action, reward, next_state):
        ''' Function for learning with Dyna. Differs from the learn-function for
        regular Q-learning as the next_state is given from the transitions in the
        T-table and that the Q-table that is used is updated for every Dyna-run'''

        # Selects the best action the next state by using the latest Q-table
        best_action = self.select_action(next_state, True, True)

        # Gets the maximum Q value of the next state
        max_Q = self.Q[next_state][best_action]

        # Updates the Q-value for the state and action with the update rule.
        # Future rewards are discounted.
        self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*(reward +  self.gamma*max_Q)

    def learn_R(self, state, action, reward):
        ''' Function for updating the R-table for with the experienced reward
        for the current state and action'''
        # Updates the expected reward for the current state with the update rule
        # without any future rewards
        self.R[state][action] = (1-self.alpha)*self.R[state][action] + self.alpha*reward

    def update_epsilon(self):
        ''' Function for decreasing the random exploration factor (epsilon) and
        increment the time variable t'''

        # Increments the time variable t by one
        self.t += 1.0

        # If the learning mode is training only (train_only) will 0epsilon be
        # decreased
        if self.learning == 'train_only':
            # Decreases epsilon exponentially
            self.epsilon = self.epsilon_start * np.exp(-self.epsilon_decay*self.t)


    def get_state(self, stock, day):
        ''' Function for retrieving the state for a stock for given day combined
        with the ownership status'''

        # Gets the state for the stock at the given day.
        state_from_data = self.states[self.learning][stock][day]

        # Gets the ownership status
        ownership_state = (self.stock_ledger[stock]['ownership'],)

        # Combines the state for the stock with the ownership status to create
        # the output state
        state = state_from_data + ownership_state
        return state

    def get_reward(self, stock, action):
        ''' Gets the reward for the executed action for the given stock'''

        # Get the daily return for the following day.
        daily_return = self.data[self.learning][stock]['Daily_return'].loc[self.next_day]

        # Gets the ownership status for the stock.
        ownership = self.stock_ledger[stock]['ownership']

        # Initializes the reward to zero
        reward = 0

        if ownership == 'Owned':
            # If the stock is owned and the selected action is sell.
            if action == 'Sell':
                # The reward is the negativ of the return. I.e if the stock goes
                # up after selling will the reward be negative.
                reward = -daily_return
            # If the stock is owned and the selected action is nothing.
            elif action == 'Nothing':
                # The reward is the return. I.e if the stock goes
                # down when keeping (not selling) the stock will it be negative.
                reward = daily_return


        elif ownership == 'Not owned':
            # If the stock is not owned and the selected action is buy.
            if action == 'Buy':
                # The reward is the return. I.e if the stock goes
                # up after buying will the reward be positive.
                reward = daily_return
            # If the stock is not owned and the selected action is nothing.
            elif action == 'Nothing':
                # The reward is the negativ of the return. I.e if the stock goes
                # up after not buying will the reward be negative.
                reward = -daily_return

        return reward


    def logger(self,reward,stock,action,state):
        ''' Function to log values when running the Q-learner. The log is kept
        in a dictionary containing lists'''

        self.log['t'].append(self.t)
        self.log['epsilon'].append(self.epsilon)
        self.log['alpha'].append(self.alpha)
        self.log['reward'].append(reward)
        self.log['state'].append(state)
        self.log['action'].append(action)
        self.log['stock'].append(stock)
        self.log['balance'].append(self.total_balance())
        self.log['train'].append(self.learning)


    def check_date(self,stock,day):
        ''' Function used for checking the stock was traded for a given date'''
        first_day = self.trading_days[stock][0]
        last_day = self.trading_days[stock][1]
        if day >= first_day and day<= last_day:
            return True
        else:
            return False

    def total_balance(self):
        ''' Function to check the total balance for all stocks in for the ledger'''
        balance = 0
        # Loops through all stocks and adds them to the balance variable.
        for stock in self.stock_ledger:
            balance += self.stock_ledger[stock]['balance']
        return balance

    def most_probable_next_state(self, state, action):
        ''' Function to get the next state that is most likely from the T-table
        when running Dyna.'''

        # Gets a list of next states for the current state and action
        states = self.T[state][action].keys()

        # Gets a list of values for the next states for the current state and
        # and action. The values are the number recorded transitions
        values = self.T[state][action].values()

        # Initializes the max_value and the max index number to zero.
        max_value = 0
        max_index = 0

        # Loops through the value list to decided what next state is most probable
        for n, value in enumerate(values):
            # If the value is larger than the max_value will the max_value and
            # the max_index be updated to the value and n
            if value > max_value:
                max_value = value
                max_index = n
            # If the max_value is equal to the current value will one be selected randomly.
            elif value == max_value:
                p = random.random()
                if p >= 0.5:
                    max_value = value
                    max_index = n

        next_state = states[max_index]
        return next_state

    def get_random_state(self):
        ''' Function that outputs a random state'''

        # Gets a list of all possible states
        states = self.Q.keys()

        # Gets the number of different states
        num_states = len(states)

        # Takes a random number in the range of the number of states
        random_num = random.randint(0,num_states-1)

        # Gets what state is at that random number
        random_state = states[random_num]

        return random_state

    def get_random_action(self,state):
        ''' Function that outputs a random action for a given state. '''

        # Gets a list of all possible action for the state
        actions = self.Q[state].keys()

        # Gets the number possible actions
        num_actions = len(actions)

        # Gets a random number in the range of the number of actions
        random_num = random.randint(0,num_actions-1)

        # Gets the action from the list at the random number.
        random_action = actions[random_num]

        return random_action


    def run_dyna(self, number_of_runs):
        ''' Function to run a set number of Dyna simulations'''

        # Loops through 0 to the number of set runs
        for n in range(0,number_of_runs,1):

            # Gets a random state
            random_state = self.get_random_state()

            # Gets a random action from the random state
            random_action = self.get_random_action(random_state)

            # Get the most likely next state from the T-table
            likley_next_state = self.most_probable_next_state(random_state, random_action)

            # Gets the expected reward from the R-table
            likley_reward = self.R[random_state][random_action]

            # Updates the Q-table with Dyna-learning
            self.learn_dyna(random_state, random_action, likley_reward, likley_next_state)

    def update(self, stock):
        '''Function to run a cycle of Q-learning for a given stock'''

        # Gets the current state for the stock
        state = self.get_state(stock,self.today)

        # Selects the best action for the state
        action = self.select_action(state, False, False)

        # Gets the reward from executing the action
        reward = self.get_reward(stock, action)

        # Updates the stock ledger for the stock with the action
        self.update_ledger(stock,action)

        # Updates the Q-table with the update rule
        self.learn(state,action,reward, stock)

        # Increments the T-table for the state and action
        self.increment_T(stock,state,action)

        # Updates the R-table with the experienced reward
        self.learn_R(state,action,reward)

        # Logs the current actions, variables and results
        self.logger(reward,stock,action,state)

        # Updates epsilon and increments the time variable t.
        self.update_epsilon()

    def check_benchmark(self):
        ''' Function for check how the Q-learner performs compared to the benchmark '''

        # Initializes a dictionary for keeping the benchmark for each stock.
        self.trading_returns = {}

        # Initializes a dictionary with list for each benchmark
        return_arrays = {'Q-learning': [], 'Benchmark': []}

        # Loops through all stocks and computes the benchmark and returns
        for stock in self.stocks:

            if stock in self.data[self.learning].keys():
                # Checks that the first day of the evaluation period is before the
                # last trading day of the stock and that the stocks trading interval
                # at least one day.
                if self.first_day < self.trading_days[stock][1] and not self.trading_days[stock][0] == self.trading_days[stock][1]:

                    # Decides what should be the start and end day for the stock data
                    # that should be evaluated.

                    # If the first day of the train and evaluation period is later or equal to
                    # the first trading day of the stock
                    if self.first_day >= self.trading_days[stock][0]:
                        # The start date of the stock for the benchmark is the first
                        # day of the evaluation period
                        start_day = self.first_day
                    else:
                        # The start date of the stock for the benchmark is the first
                        # day of the trading period of the stock
                        start_day = self.trading_days[stock][0]

                    # If the last day of the train and evaluation period is later or equal to
                    # the last trading day of the stock
                    if self.last_day >= self.trading_days[stock][1]:
                        # The end date for the stock when performing the benchmark is
                        # the last trading day
                        end_day = self.trading_days[stock][1]
                    else:
                        # The end date for the stock when performing the benchmark is
                        # the last day of the train and evaluation period.
                        end_day = self.last_day

                    # Gets the start and end prices of the stock at the start and
                    # end days.
                    start_price = self.data[self.learning][stock]['Price'][start_day]
                    end_price = self.data[self.learning][stock]['Price'][end_day]

                    # Gets the profit from running the Q-trader
                    profit = self.stock_ledger[stock]['balance']

                    # Normalizes the the return of Q-trading with the start price
                    return_from_Q = profit/start_price

                    # Computes the normalized return from the benchmark of buying and
                    # holding the stock
                    return_from_benchmark = (end_price-start_price)/end_price

                    # Stores the results for the stock in the trading_returns dictionary
                    self.trading_returns[stock] = {'Q-learning': return_from_Q, 'Benchmark': return_from_benchmark}

                    # Stores the results in array for being able to compute average
                    # results later
                    return_arrays['Q-learning'].append(return_from_Q)
                    return_arrays['Benchmark'].append(return_from_benchmark)


        # Initializes a empty dictionary for the mean returns
        self.mean_returns = {}

        # Computes the mean for each variable stored in the return arrays
        for key in return_arrays:
            self.mean_returns[key] = np.mean(np.array(return_arrays[key]))

        return self.mean_returns

    def run(self,train):
        ''' Function for running the Q-learning algorithm to perform reinforcement
         learning with Dyna. The train input is what kind of training should be
         performed. Either training only - 'train_only' or training and evaluation
         - 'train_eval' '''

        # Sets the current learning type to train setting
        self.learning = train

        # If the training mode is training and evaluation
        if train == 'train_eval':
            # Sets the random exploration factor to zero
            self.epsilon = 0.0

            # Reset the stock ledger so that it can be used for evaluation
            self.stock_ledger = self.init_stock_ledger()

        # Sets first_day from the first day in the training set
        self.first_day = self.dates[train][0]
        # Loops through all days in the selected training set
        for n, day in enumerate(self.dates[train][:-2]):
            #print day

            # Sets the current day
            self.today = day

            # Set the next day
            self.next_day = self.dates[train][n+1]

            # Set the Q-table for the current day
            self.Q_today = self.Q

            # Loops through all stocks in the training set
            for stock in self.data[train]:
                # Check is the there is data for the current stock and day
                check_today = self.check_date(stock,self.today)

                # Check is the there is data for the current stock and next day
                check_next_day = self.check_date(stock,self.next_day)

                # Is both today and next day has data
                if check_today and check_next_day:
                    # Runs update to perform Q-learning
                    self.update(stock)

                # If this is the final trading day for the stock
                elif check_today and not check_next_day:
                    # Sell the stock
                    self.update_ledger(stock,'Sell')


            # Runs a set number of Dyna-cycles
            self.run_dyna(1000)

        # Sets today to next_day when the penultimate day has been looped
        self.today = self.next_day
        # Sets last_day to the current day
        self.last_day = self.today

        # Loops through all stocks to sell them as the trading period has ended
        for stock in self.data[train]:
            if self.check_date(stock,self.today):
                self.update_ledger(stock,'Sell')


        print('Finished')
