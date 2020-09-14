import numpy as np
import pandas as pd
import torch
import math
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import *
from pandas.tseries.offsets import MonthEnd
from itertools import count
import TE_CAAN as tc
import warnings
import gc

# Turn off warnings
warnings.filterwarnings(action='ignore')

# Environment
class env:
    'Resembles Stock Market'
    def __init__(self, data, FACTORS, device):

        # Unprocessed Data
        self.data = data
        self.FACTORS = FACTORS
        self.device = device

        #Process factors
        self.processed_data = self.data[['code', 'mdate', 'ret_1m'] + FACTORS].copy(deep=True) #ret_1m will be excluded later
        self.processed_data.sort_values(['code', 'mdate'], inplace=True)
        self.processed_data.reset_index(drop=True, inplace=True)

        #Creat list of dates that will be considered in the model
        self.datelist = [pd.Timestamp(item) for item in list(self.processed_data['mdate'].unique())]

    # Select previous 12months and transform data into tensors
    def encoder_12m(self, date):  # EX: When date is 2020-01, 2019-01 ~ 2019-12 will be returned

        #Calculate the start date of 12months
        mdate_start = date - MonthEnd(12)

        #Slice data by date
        data_12m = self.processed_data[(self.processed_data['mdate'] >= mdate_start)
                                       & (self.processed_data['mdate'] <= date)].copy(deep=True)
                                        #include last month for now and will be excluded later
                                        #We do this process to eliminate data of which ret_1m cannot be calculated

        #Try to count number of existing months per firm
        data_12m['nobs'] = data_12m.groupby(['code'])['mdate'].transform('count')

        if date <= (self.last_date() + relativedelta(day=31)) : #If we need return information:
            #Take firms that have 13months of data
            data_12m = data_12m[data_12m['nobs'] == 13].copy(deep=True)

        else: #If we don't need return information
            #Take firms that have 12months of data
            data_12m = data_12m[data_12m['nobs'] == 12].copy(deep=True)

        # Exclude 13th month
        data_12m = data_12m[data_12m['mdate'] < date].copy(deep=True)
        data_12m.sort_values(['code', 'mdate'], inplace=True)
        data_12m.reset_index(drop=True, inplace=True)
        data_12m.drop(columns=['nobs'], inplace=True)

        #Define key table
        key_table = pd.Series(data_12m['code'].unique())


        # Vectorize the input data

        three_dim = np.array(data_12m).reshape(-1, 12, data_12m.shape[1])  # (1624, 12, 22)

        #Here we exclude code, mdate, ret_1m from the input data
        three_dim_tensor = torch.from_numpy(three_dim[:, :, 3:].astype(float))  # [Number of Firm, Time, Features]
        three_dim_tensor = three_dim_tensor.to(self.device)

        return three_dim_tensor, key_table


    #Define Useful Methods from now on
    def random_state(self, train_begin, train_end, seed_num):
        'Draw a random state as a form of tensor'
        #list of dates that we can choose from
        date_choice = [item for item in self.datelist if
                       item >= train_begin + MonthEnd(12) and item <= train_end - MonthEnd(11)]

        #Set a seed_number
        #np.random.seed(seed_num)

        #draw random a random date
        random_date = np.random.choice(date_choice)
        return random_date, self.encoder_12m(random_date)


    def next_state(self, current_date):
        'Return next_state as a form of tensor'
        next_month = current_date + MonthEnd(1)  # next_month
        return next_month, self.encoder_12m(next_month)


    def portfolio_return(self, action, stocks, current_date):
        'Calculate portfolio_return given state and action'

        #Take the processed data of current_date
        current_state = self.processed_data[self.processed_data['mdate'] == current_date].copy(deep=True)
        current_state = current_state.set_index('code')
        current_state = current_state.loc[stocks]

        # Calculate Portfolio Return
        return_tensor = torch.tensor(current_state['ret_1m'], device=self.device) #convert return into tensor
        action_tensor = action.squeeze(1) #dimension reduction

        # multiply two tensors and add them to calculate total return
        port_return = (return_tensor * action_tensor).sum().reshape(1)

        return port_return


    def last_date(self):
        'return last date in the environmnet'
        return self.datelist[-1]


#Define What Happens in an Episode
def run_episode(policy, market, optimizer, scheduler, train_flag, **kwargs):
    global date, state

    # create empty lists to store action, stock, and date
    stock_list = []
    action_list = []
    date_list = []

    # Input_parameters
    train_begin = kwargs.get('train_begin', None)
    train_end = kwargs.get('train_end', None)
    epoch = kwargs.get('epoch', None)

    # initial state random selection
    if train_flag == True:

        #for train, we need to start with random state

        #Set seed_number for the sake of train analysis
        seed_num = epoch
        date, (state, key_table) = market.random_state(train_begin, train_end, seed_num = seed_num)

        #Pick actions based state and policy
        action, stocks = policy(state, key_table=key_table)

        #Calculate returns given stocks and weights
        reward = market.portfolio_return(action, stocks, date)

        # Store Information
        policy.rewards.append(reward)  # Store rewards
        stock_list.extend(stocks.to_list())  # store stock
        action_list.extend(action.cpu().detach().squeeze().tolist())  # store action
        date_list.extend([datetime.strftime(date, '%Y-%m')] * policy.threshold)  # store dates when action is taken

        # remaining rounds for training
        remaining = 11

    elif train_flag == False: #If we are running test rounds
        # for test, we just iterate 12times since we do not need random states
        remaining = 12  # remaining rounds for test
    else:
        print("No flag indicated")

    # Run remaining rounds. Total 12 in one episode
    for _ in range(remaining):
        #Take next state
        date, (state, key_table) = market.next_state(date)

        #Pick actions based state and policy
        action, stocks= policy(state, key_table=key_table)

        # Calculate returns given stocks and weights
        reward = market.portfolio_return(action, stocks, date)

        #Store Information
        policy.rewards.append(reward)  # Store rewards
        stock_list.extend(stocks.to_list())  # store stock
        action_list.extend(action.cpu().detach().squeeze().tolist())  # store action
        date_list.extend([datetime.strftime(date, '%Y-%m')] * policy.threshold)  # store dates when action is taken

    #Calculating our objective function: Sharpe Ratio
    rewards = torch.cat(policy.rewards)
    sharpe_ratio = torch.mean(rewards) / torch.std(rewards)  # without risk-free rate
    loss = -sharpe_ratio

    #Optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Print current epoch and Sharpe Ratio
    sharpe_ratio_item = round(sharpe_ratio.item(), 2)
    if train_flag:
        print('Epoch : ',epoch+1, '\t Sharpe Ratio : ',sharpe_ratio_item)

    else:
        print(f'{datetime.strftime(date - MonthEnd(11), "%Y-%m")} : '
              f'{datetime.strftime(date, "%Y-%m")}\t Sharpe Ratio : {sharpe_ratio_item}')

    # Save values for future reference
    cum_ret = torch.cumprod(torch.add(rewards.data, 1), dim=0)[-1].item()
    avg_ret = torch.mean(rewards.data).item()
    sharpe_ratio = sharpe_ratio.item()

    # Clean variables that will no longer be used
    del policy.rewards[:]
    del rewards
    del loss

    # Clear Memory
    gc.collect()
    torch.cuda.empty_cache()

    # Return current episode output
    return stock_list, action_list, avg_ret, cum_ret, sharpe_ratio, date_list


# Train
def train(policy, market, optimizer, scheduler, episode_num, train_begin, train_end):
    print('Train Begins')

    train_port = pd.DataFrame()
    train_log = pd.DataFrame()

    # repeat episode
    for epoch in range(episode_num):
        # run episode
        stock_code, weights, avg_ret, cum_ret, sr, dl = run_episode(policy, market, optimizer, scheduler,
                                                           train_flag=True, train_begin=train_begin,
                                                           train_end=train_end, epoch=epoch)

        # episode start date and end date
        ep_start_date = datetime.strftime(date - relativedelta(months=11), '%Y-%m')
        ep_end_date = datetime.strftime(date, '%Y-%m')

        # Train_port report prep
        temp_port = pd.DataFrame()
        temp_port['Stock_Codes'] = stock_code
        temp_port['Weights'] = weights
        temp_port['Invest_Date'] = dl
        temp_port.set_index('Invest_Date', inplace=True)

        # Concat new report
        train_port = pd.concat([train_port, temp_port], axis=0)
        del temp_port

        # Train_log report prep
        index = f'{ep_start_date}:{ep_end_date}'

        # Record Average return
        if index in train_log.index:
            train_log.at[index, f'Average_Return_{int(train_log.loc[index].notnull().sum() / 3) + 1}'] = avg_ret
        else:
            train_log.at[index, 'Average_Return_1'] = avg_ret

        #Record Cumulative Return
        if train_log.loc[index].notnull().sum() == 1:
            train_log.at[index, 'Cumulative_Return_1'] = cum_ret
        else:
            train_log.at[index, f'Cumulative_Return_{int((train_log.loc[index].notnull().sum() + 2 ) / 3)}'] = cum_ret

        # Record Sharpe Ratio
        if train_log.loc[index].notnull().sum() == 2:
            train_log.at[index, 'Sharpe_Ratio_1'] = sr
        else:
            train_log.at[index, f'Sharpe_Ratio_{int((train_log.loc[index].notnull().sum() + 1) / 3)}'] = sr

    # Name index of train_log
    train_log.index.name = 'Investment Period'
    train_log.sort_index(axis=0, inplace=True)

    return train_log, train_port


# Test
def test(policy, market, optimizer, scheduler, test_begin):
    global date

    print('Test Begins')

    test_log = pd.DataFrame()
    test_port = pd.DataFrame()

    # subtract 1 month so that we can use next_state method directly for test_begin
    date = test_begin - MonthEnd(1)

    # Initiate test iteration
    for _ in count(1):

        # run episode
        stock_code, weights, avg_ret, cum_ret, sr, dl = run_episode(policy, market, optimizer, scheduler, train_flag=False)

        # episode start date and end date
        ep_start_date = datetime.strftime(date - MonthEnd(11), '%Y-%m')
        ep_end_date = datetime.strftime(date, '%Y-%m')

        # Test_port report prep
        temp_port = pd.DataFrame()
        temp_port['Stock_Codes'] = stock_code
        temp_port['Weights'] = weights
        temp_port['Invest_Date'] = dl
        temp_port.set_index('Invest_Date', inplace=True)

        # Concat new report
        test_port = pd.concat([test_port, temp_port], axis=0)
        del temp_port

        # Test_log report prep
        test_log.at[f'{ep_start_date}:{ep_end_date}', 'Average_Return'] = avg_ret
        test_log.at[f'{ep_start_date}:{ep_end_date}', 'Cumulative_Return'] = cum_ret
        test_log.at[f'{ep_start_date}:{ep_end_date}', 'Sharpe_Ratio'] = sr

        # Calculate episode stop date(subtract 12 month + 1 month)
        stop_date = market.last_date() - relativedelta(months=13) + relativedelta(day=1)

        # Stop episode on a specific date
        if date >= stop_date:
            print(f'last episode end date: {datetime.strftime(date, "%Y-%m")}')
            break

        # Name index of test_log
        test_log.index.name = 'Investment Period'

    return test_log, test_port


# Save Parameters of TE-CAAN:
def save_param(path, policy):
    torch.save(policy, path)


# Load Parameters of TE-CAAN:
def load_model(path, eval_flag):
    #Load Model
    policy = torch.load(path)

    if eval_flag == True:
        policy.eval()

    elif eval_flag == False:
        policy.train()

    else:
        ('Please specify evaluation flag')

    return policy


# Check Parameters of TE-CAAN
def check_param(policy):
    for param_tensor in policy.state_dict():
        print(param_tensor, "\t", policy.state_dict()[param_tensor].size())


# Train Data Analysis Tool
def fig_analysis(data_path, what_to_analyze):
    data = pd.read_csv(data_path)
    data = data.set_index('Investment Period')
    data = data.loc[:, data.columns.str.startswith(f'{what_to_analyze}')]
    data.columns = range(1, len(data.columns) + 1)
    period_list = data.index
    length = data.index.shape[0]
    fig, axs = plt.subplots(math.ceil(length / 5), 5,
                            figsize=((math.ceil(length) / 5) * 11, (math.ceil(length / 5)) * 11)) #sharey=True)
    fig.text(0.5, 0.006, 'Iterations', ha='center', va='center', fontsize=100)
    fig.text(0.09, 0.5, 'Values', ha='center', va='center', rotation='vertical', fontsize=100)
    fig.suptitle(f'{what_to_analyze}', fontsize=100)
    plt.subplots_adjust(top=0.95, bottom=0.02, hspace=0.4, wspace=0.2)
    plt.rcParams['xtick.labelsize'] = 50
    plt.rcParams['ytick.labelsize'] = 50
    for i, period in enumerate(period_list):
        plot_data = data.loc[period].dropna()
        axs[i // 5, i % 5].plot(plot_data, lw=10)
        axs[i // 5, i % 5].set_title(f'{period}', fontsize=60)
    plt.savefig(f'Train_Analysis_{what_to_analyze}.jpg')


# Create investment portfolio using trained model
def select_portfolio(data_path, model_path, invest_date, FACTORS, device):
    # Input should include date of investment (not the last date of available data set)

    # Set empty dataframe for report
    investment_portfolio = pd.DataFrame()

    # need to set device_type
    device_type = device  # 'cpu' or 'cuda'
    device = torch.device(device_type)

    # Date to datetime format
    date = datetime.strptime(invest_date, '%Y-%m-%d') + relativedelta(day=31)

    # Load Saved Parameters
    policy = load_model(model_path, eval_flag=True)

    # Create invest_port
    data = pd.read_pickle(data_path)

    #Preprocess Data

    # Preprocessing - Rank Transform
    data[FACTORS] = (data.groupby(['mdate'])[FACTORS].rank(pct=True) - 0.5).fillna(0)
    data.sort_values(['code', 'mdate'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Preprocessing - Remove rows that do not have return information
    data = data.loc[data['ret_1m'].notna()]

    # Excluding Microcaps increase performance
    data = data[data['size'] >= -.3].copy(deep=True)

    market = env(data, FACTORS, device)
    state, key_table = market.encoder_12m(date)
    action, stocks = policy(state, key_table=key_table)

    #action into list
    action_list = action.cpu().detach().squeeze().tolist()

    #stocks into list
    stock_list = stocks.to_list()

    # Investment portfolio preparation
    investment_portfolio['Stock_Codes'] = stock_list
    investment_portfolio['Weights'] = action_list
    investment_portfolio['Invest_Date'] = invest_date
    investment_portfolio.set_index('Invest_Date', inplace=True)

    # Save investment_portfolio
    investment_portfolio.to_csv(f'./{invest_date}_portfolio_selection.csv')

    print('Portfolio Selected. Check CSV file')

    return investment_portfolio
