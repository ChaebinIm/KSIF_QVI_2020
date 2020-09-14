import Alphaportfolio as ap
import TE_CAAN as tc
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from pandas.tseries.offsets import MonthEnd


def main(episode_num):
    # Train Model

    FACTORS = ['e_p', 'b_p', 's_p', 'gp_p', 'op_p', 'c_p', 'roa', 'roe', 'roic', 'gp_a', 'gp_s', 'salesqoq',
               'gpqoq', 'roaqoq', 'prior_2_6', 'prior_2_12', 'liq_ratio','equity_ratio', 'debt_ratio',
               'foreign_ownership_ratio', 'vol_1m', 'size']

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda'  # 'cpu' or 'cuda'
    device = torch.device(device_type)

    # Data Load & Preprocessing
    data = pd.read_pickle('data/2020-08-28_company.pck')

    #Preprocessing - Rank Transform
    data[FACTORS] = (data.groupby(['mdate'])[FACTORS].rank(pct=True)-0.5).fillna(0)
    data.sort_values(['code', 'mdate'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    #Preprocessing - Remove rows that do not have return information
    data = data.loc[data['ret_1m'].notna()]

    # Excluding Microcaps increase performance
    data = data[data['size'] >= -.3].copy(deep=True)

    # Create An Environment
    market = ap.env(data, FACTORS, device)

    # Define Policy
    # Set initial random values
    column_num = len(market.processed_data.columns) - 3
    threshold = 20

    # print(company_num, column_num,)

    policy = tc.Transformer(d_model=88, nhead=4, nhead2=1,  # The paper suggests nhead value of 4
                            num_encoder_layers=1, column_num=column_num,
                            threshold=threshold).to(device)

    # Train

    # Optimizer for train
    train_optimizer = optim.Adam(policy.parameters(), lr=0.00002)  #For 500 stocks => 0.005, For 20 Stocks => 0.003
    train_scheduler = StepLR(train_optimizer, step_size=500, gamma=0.95) #For 500 stocks => Step_size = 10, For 20 Stocks = > Step_size 100

    # Run Train
    train_begin = pd.Timestamp('2003-01-31')
    train_end = pd.Timestamp('2011-07-31')

    #Set policy to train mode
    policy.train()

    #Run train
    train_log_reward, train_log_port = ap.train(policy=policy, market=market,
                                                optimizer=train_optimizer, scheduler=train_scheduler,
                                                episode_num=episode_num, train_begin=train_begin, train_end=train_end)

    # Save Train Results as CSV
    train_log_reward.to_csv('./train_log_reward.csv')
    train_log_port.to_csv('./train_log_port.csv')

    # Create result graph
#    ap.fig_analysis('./train_log_reward.csv', 'Average_Return')
#    ap.fig_analysis('./train_log_reward.csv', 'Sharpe_Ratio')
#    ap.fig_analysis('./train_log_reward.csv', 'Cumulative_Return')

    # Save model for train_set
#    ap.save_param('./train_set.pt', policy)


    #Test

    # Optimizer for Test
    test_optimizer = optim.Adam(policy.parameters(), lr=0)
    test_scheduler = StepLR(test_optimizer, step_size=100, gamma=0.95)


    # Set best begin date
    test_begin = pd.Timestamp('2011-08-01') + MonthEnd(0)

    #Set policy to evaluation mode
    policy.eval()

    #Run Test
    test_log_reward, test_log_port = ap.test(policy=policy, market=market, optimizer=test_optimizer,
                                             scheduler=test_scheduler, test_begin=test_begin)

    # Save Test Results as CSV
    test_log_reward.to_csv('./test_log_reward.csv')
    test_log_port.to_csv('./test_log_port.csv')

    # Save model for test_set
#    ap.save_param('./test_set.pt', policy)


if __name__ == "__main__":
    main()