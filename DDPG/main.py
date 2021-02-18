import numpy as np
from environment import Agent
import pickle
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from pandas.tseries.offsets import MonthEnd
from itertools import product

if __name__ == '__main__':

    run_type = 'test' # 'test' or 'train'

    FACTORS = ['e_p', 'b_p', 's_p', 'gp_p', 'op_p', 'c_p', 'roa', 'roe', 'roic', 'gp_a', 'gp_s', 'salesqoq',
               'gpqoq', 'roaqoq', 'prior_2_6', 'prior_2_12', 'liq_ratio', 'equity_ratio', 'debt_ratio',
               'foreign_ownership_ratio', 'vol_1m', 'size']

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cpu'  # 'cpu' or 'cuda'
    device = torch.device(device_type)

    # Data Load & Preprocessing
    data = pd.read_pickle('data/2020-11-27_company.pck')
    data['mdate'] = pd.to_datetime(data['mdate'])
    reward_list = data[['code','mdate','sharpe_1m_LEAD_1']]

    dates = data['mdate'].unique()

    # devide dates into train, validation, test
    train_start_date = pd.to_datetime(dates[0])
    train_end_date =  pd.to_datetime('2015-01-31')
    valid_end_date =  pd.to_datetime('2017-01-31')
    test_end_date = pd.to_datetime('2020-10-31')


    # Create An Environment
#    agent = Agent(data, FACTORS, device,
#                  input_dims=(500,12,22) , lr = 0.0001,
#                  batch_size=1,
#                  n_actions= 20 ) # threshold = 20



    ####
    #train start : define hyperparameters(epoch, gamma, tau, lr1, lr2)
    # epoch : [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
    # gamma : [0.90,0.92,0.94,0.96,0.98]
    # tau : [0.01,0.05,0.1,0.2,0.3]
    # lr1 : [0.00001, 0.0001, 0.001, 0.01, 0.1]
    # lr2 : [0.00001, 0.0001, 0.001, 0.01, 0.1]

    # 0.95,0.96,0.97, 0.98, 0.99, 0.995
    parameter_set = [[1000], [0.95,0.96,0.97],
                     [0.01, 0.05, 0.1], [0.00001, 0.0001, 0.001, 0.01],
                     [0.00001, 0.0001, 0.001, 0.01]]
    parameter_set_frame = pd.DataFrame(list(product(*parameter_set)), columns=['epoch', 'gamma','tau','lr1', 'lr2'])

    # train & validation
    if run_type == 'train':
        #agent.train(train_start_date, train_date, epoch, gamma = 0.99, tau = 0.01, save_path = './test_01.pt')
        for i in range(len(parameter_set_frame)):
            epoch_grid = int(parameter_set_frame.iloc[i]['epoch'])
            gamma_grid = parameter_set_frame.iloc[i]['gamma']
            tau_grid = parameter_set_frame.iloc[i]['tau']
            lr1_grid = parameter_set_frame.iloc[i]['lr1']
            lr2_grid = parameter_set_frame.iloc[i]['lr2']
            save_path = './model/'+str(epoch_grid) + '_' + str(gamma_grid) + '_' + str(tau_grid) + '_' + str(lr1_grid)+'_'+str(lr2_grid)+'.pt'

            a = Agent(data, FACTORS, device,
                      input_dims=(500,12,22) , lr1 = lr1_grid, lr2 = lr2_grid,
                      batch_size=1, n_actions= 20)
            a.train(train_start_date, train_end_date, epoch_grid, gamma_grid, tau_grid, save_path)
            a.validation(train_end_date, valid_end_date, epoch_grid, gamma_grid, tau_grid)
    elif run_type == 'test':
        for i in range(len(parameter_set_frame)):
            epoch_grid = int(parameter_set_frame.iloc[i]['epoch'])
            gamma_grid = parameter_set_frame.iloc[i]['gamma']
            tau_grid = parameter_set_frame.iloc[i]['tau']
            lr1_grid = parameter_set_frame.iloc[i]['lr1']
            lr2_grid = parameter_set_frame.iloc[i]['lr2']
            save_path = './model/' + str(epoch_grid) + '_' + str(gamma_grid) + '_' + str(tau_grid) + '_' + str(
                lr1_grid) + '_' + str(lr2_grid) + '.pt'
            tmp = Agent(data, FACTORS, device,
                      input_dims=(500, 12, 22), lr1=lr1_grid, lr2=lr2_grid,
                      batch_size=1, n_actions=20)
            optimal_model = torch.load(save_path)
            tmp.test(optimal_model, valid_end_date, test_end_date, epoch_grid, gamma_grid, tau_grid)












    def make_critic_state(_state, _stocks):
        stock_index_ls = _stocks.index.to_list()
        critic_state = _state[[stock_index_ls]]

        return critic_state

