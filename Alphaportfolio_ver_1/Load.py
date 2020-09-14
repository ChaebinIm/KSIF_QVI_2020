import Alphaportfolio as ap
import torch

#Load
device_type = 'cuda'  # 'cpu' or 'cuda'
device = torch.device(device_type)

#Load Trained Parameters
policy = ap.load_model(path = './test_set.pt', eval_flag = True)

#Check Parameters of Loaded Parameters
ap.check_param(policy)

#Select Portfolio
FACTORS = ['e_p', 'b_p', 's_p', 'gp_p', 'op_p', 'c_p', 'roa', 'roe', 'roic', 'gp_a', 'gp_s', 'salesqoq',
           'gpqoq', 'roaqoq', 'prior_2_6', 'prior_2_12', 'liq_ratio', 'equity_ratio', 'debt_ratio',
           'foreign_ownership_ratio', 'vol_1m', 'size']

ap.select_portfolio(data_path='data/2020-08-28_company.pck', model_path='./test_set.pt', invest_date='2020-09-01',
                    FACTORS=FACTORS, device='cuda')

