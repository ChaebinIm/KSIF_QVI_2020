import numpy as np
import pandas as pd
import torch
import torch as T
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import *
from pandas.tseries.offsets import MonthEnd
# from tqdm import tqdm
from torch.autograd import Variable
from itertools import count
import warnings
import gc
from buffer import ReplayBuffer
from Actor_TE_CAAN import Transformer as ActorNetwork
from Critic_TE_CAAN import CriticNetwork as CriticNetwork
# Turn off warnings
warnings.filterwarnings(action='ignore')

#actor class 에 noise 주기위한 함수들
class OUActionNoise():
    def __init__(self, mu, sigma=0.01, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)





# Environment
class Agent():
    def __init__(self, data, FACTORS, device, input_dims,  n_actions = 20, lr1=0.0001, lr2=0.0001, max_size=100, batch_size=1): #, gamma=0.99,   # alpha, beta tau


        # Unprocessed Data
        self.data = data
        self.FACTORS = FACTORS
        self.device = device
        self.lr1 = lr1
        self.lr2 = lr2
        self.reward_list = self.data[['code', 'mdate', 'sharpe_1m_LEAD_1', 'ret_1m']]

        # Process factors
        self.processed_data = self.data[['code', 'mdate', 'ret_1m'] + FACTORS].copy(
            deep=True)  # ret_1m will be excluded later
        self.processed_data.sort_values(['code', 'mdate'], inplace=True)
        self.processed_data.reset_index(drop=True, inplace=True)
        self.standard_column = 'size'
        # Creat list of dates that will be considered in the model
        self.datelist = [pd.Timestamp(item) for item in list(self.processed_data['mdate'].unique())]

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

      #  self.gamma = gamma
      #  self.tau = tau
        self.batch_size = batch_size
        # self.alpha = alpha
        # self.beta = beta

        self.replay_buffer = ReplayBuffer(max_size, input_dims, n_actions)

       # self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(d_model=88, nhead=4, nhead2=1,  # The paper suggests nhead value of 4
                                 num_encoder_layers=1, column_num=len(self.FACTORS), lr = self.lr1,
                                  threshold=n_actions).to(device)

        self.critic = CriticNetwork(d_model=88, nhead=4, nhead2=1,  # The paper suggests nhead value of 4
                                 num_encoder_layers=1, column_num=len(self.FACTORS), lr = self.lr2,
                                  threshold=n_actions).to(device)

# off policy 라서 target 과 behavior actor critic 이나뉨





        # 네트워크 조정 후 코드 수정해야됨.
        self.target_actor = ActorNetwork(d_model=88, nhead=4, nhead2=1,  # The paper suggests nhead value of 4
                                 num_encoder_layers=1, column_num=len(self.FACTORS), lr = lr1,
                                  threshold=n_actions).to(device)

        self.target_critic =  CriticNetwork(d_model=88, nhead=4, nhead2=1,  # The paper suggests nhead value of 4
                                 num_encoder_layers=1, column_num=len(self.FACTORS), lr = lr2,
                                  threshold=n_actions).to(device)









        # target 과 behavior network를 각각 업데이트 하기위해서 함수를 만들었음. tau는 behavior의 비율임 1에서 0으로 갈수록
        #target network 업데이트시 behavior network의 비율이 작아짐.
        self.update_network_parameters(tau=1)


    def normaliztion(self, data):
        data[self.FACTORS] = ( data.groupby(['mdate'])[self.FACTORS].rank(pct=True) - 0.5).fillna(0)
        data.sort_values(['code', 'mdate'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        data =  data.loc[data['ret_1m'].notna()]
        return(data)

    def make_company_table(self):
        self.company_table_per_month = {}
        for i in self.datelist[12:]:

            #### t~t-13까지 모든 데이터가 존재하는 기업을 필터링하는 조건을 걸어놓을것
            #mdate_start = i - MonthEnd(12)
            mdate_start =  self.datelist[self.datelist.index(i) - 12]

            # Slice data by date
            data_12m = self.processed_data[(self.processed_data['mdate'] >= mdate_start) & (self.processed_data['mdate'] <= i)].copy()

            # t-12 ~ t시점까지 13개월 모두 존재하는 종목 코드 filtering한 것.
            data_12m = data_12m.dropna()

            all_month_exist_company = data_12m['code'].value_counts().index[data_12m['code'].value_counts().values == 13]

            #data_12m = data_12m[data_12m['code'].isin(all_month_exist_company)]

            temp = self.processed_data[self.processed_data['mdate'] == (i-MonthEnd(1))]
            temp = temp[temp['code'].isin(all_month_exist_company)]
            ###3


            ###
            temp = temp.sort_values(self.standard_column, ascending = False)


            temp = temp.iloc[:500].reset_index(drop = True)

            self.company_table_per_month[i] = temp['code']



            # test = pd.DataFrame(self.company_table_per_month)
            # for j in range(len(test.columns.to_list())):
            #     print(test.columns.to_list()[j])
            #     print('length :', len(test.iloc[:,j]))


    def split_data_by_date(self, begin_date, end_date):
        self.split_data = self.processed_data[(self.processed_data['mdate'] < end_date) & (self.processed_data['mdate'] >= begin_date)]





        # Select previous 12months and transform data into tensors
    def encoder_12m(self, date, company_filtered):  # EX: When date is 2020-01, 2019-01 ~ 2019-12 will be returned
        self.company_filtered = company_filtered
        # self.split_data = self.split_data[self.split_data['code'].isin(self.company_filtered.values)]
        # self.split_data = self.normaliztion(self.split_data)
        # # Calculate the start date of 12months
        # # mdate_start = date - MonthEnd(12)
        # mdate_start = self.datelist[self.datelist.index(date) - 12]
        # # Slice data by date
        # data_12m = self.split_data[(self.split_data['mdate'] >= mdate_start)
        #                            & (self.split_data['mdate'] <= date)].copy(deep=True)
        self.temp_data = self.processed_data[self.processed_data['code'].isin(self.company_filtered.values)]
        self.temp_data.fillna(0, inplace = True)
        # 정규화가 shape에 영향을 미침. (단, NaN값이 많이 생김) 2020.11.26
        # 이유는 185번줄부터 무슨 이상을 줄 것으로 추정됨. 우리가 짠 코드가 아니라서 한번 자세히 봐야될듯.
        # self.processed_data = self.normaliztion(self.processed_data)
       # print(self.processed_data.shape)
        #print(self.processed_data.isna().sum())
        #print(self.normaliztion(self.processed_data).shape)
        #self.processed_data.to_excel('./searching_NA.xlsx')

        # Calculate the start date of 12months
        # mdate_start = date - MonthEnd(12)
        mdate_start = self.datelist[self.datelist.index(date) - 12]
        # Slice data by date
        data_12m = self.temp_data[(self.temp_data['mdate'] >= mdate_start)
                                   & (self.temp_data['mdate'] < date)].copy(deep=True)
        # include last month for now and will be excluded later
        # We do this process to eliminate data of which ret_1m cannot be calculated
        # Try to count number of existing months per firm
        data_12m['nobs'] = data_12m.groupby(['code'])['mdate'].transform('count')

   #     if date <= (self.datelist[-1] + relativedelta(day=31)):  # If we need return information:
    #        # Take firms that have 13months of data
        #data_12m = data_12m[data_12m['nobs'] == 13].copy(deep=True)

 #       else:  # If we don't need return information
            # Take firms that have 12months of data
  #          data_12m = data_12m[data_12m['nobs'] == 12].copy(deep=True)
        # Exclude 13th month
        data_12m = data_12m[data_12m['mdate'] < date].copy(deep=True)
        data_12m.sort_values(['code', 'mdate'], inplace=True)
        data_12m.reset_index(drop=True, inplace=True)
        data_12m.drop(columns=['nobs'], inplace=True)

        # Define key table
        key_table = pd.Series(data_12m['code'].unique())

        # Vectorize the input data

        three_dim = np.array(data_12m).reshape(-1, 12, data_12m.shape[1])  # (1624, 12, 22)
        # Here we exclude code, mdate, ret_1m from the input data
        three_dim_tensor = torch.from_numpy(three_dim[:, :, 3:].astype(float))#.to(torch.float32) # [Number of Firm, Time, Features]
        three_dim_tensor = Variable(three_dim_tensor, requires_grad=True)
        three_dim_tensor = three_dim_tensor.to(self.device)

        return three_dim_tensor, key_table



    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(),
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, idxs, codes):

        self.replay_buffer.store_transition(state, action.detach().numpy(),
                                            reward, state_, idxs, codes)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()

        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self, key_table, tau = 0.01, gamma = 0.99):
        if self.replay_buffer.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, idxs, codes   = \
                self.replay_buffer.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float)#.to(self.actor.device)
        new_states = T.tensor(new_states, dtype=T.float)#.to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float)#.to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float)#.to(self.actor.device)
      #  done = T.tensor(done).to(self.actor.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        actions, codes, idxs = self.actor.forward(states, key_table=key_table)
        # actions = actions + T.tensor(self.noise(), dtype=T.float)# .to(self.actor.device) # add noise term
        target_actions, target_codes, target_idxs = self.target_actor.forward(new_states, key_table = key_table)

        # stock_index_ls = list(codes)
        # critic_next_state_input = new_states[[stock_index_ls]]
        critic_next_state_input = new_states[target_idxs]
        critic_state_input = states[idxs]
        ############3
        # critic input으로 들어가는 new state, state등은 state를 actor에 넣어서 나오게된 회사들(action)만 추출하여
        #  critic input으로 넣을것
##################################################################
        critic_value_ = self.target_critic.forward(critic_next_state_input, target_actions)

        critic_value = self.critic.forward(critic_state_input, actions)

        #critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + gamma * critic_value_
        target = target.view(self.batch_size, 1)

      #  self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        _action, _, _ = self.actor.forward(states, key_table = key_table)

##############################################################################################
        """
            Converge가 되는지 어떻게 파악?
            weight * sharpe_ratio이 epoch가 증가함에 따라 올라가는지를 보면 된다!
        """


###############################################################################################
        # input을 어떻게 해야 되는지 고려 필요.
        actor_loss = -self.critic.forward(critic_state_input, _action)








        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters(tau)

        return critic_loss.item(), actor_loss.item()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()

        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)

        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)



        # behavior actor/critic update :: 최종 파라미터 업데이트
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()


        # behavior actor/critic에 업데이트된 내용들을 target에 loading :: 최종 업데이트된 파라미터 씌우기
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)


    # input parameter : begin_date(), end_date(), epoch(), lr(), gamma(), tau(), save_name
    def train(self, begin_date, end_date, epoch, gamma, tau, save_path):

        train_critic_loss = np.zeros(epoch)
        train_actor_loss = np.zeros(epoch)

        #self.split_data_by_date(begin_date, end_date)

        # 모든 기간의 각 월별 시총 500위 기업들을 모은 table set 생성
        self.make_company_table()

        date_choice = [item for item in self.datelist if
                       item <= end_date - MonthEnd(1) and item >= begin_date + MonthEnd(12)]

        for i in range(epoch):
            print(i)

            random_date = np.random.choice(date_choice[:-1]) # 2012-04-30
            #random_date = pd.to_datetime('2005-09-30')
            next_date = date_choice[np.where(random_date == np.array(date_choice))[0][0] + 1] # 2012-05-31

            companies_by_R_now = self.company_table_per_month[next_date] # 2011-05-31 ~ 2012-05-31 존재 유무 체크 후에 2012-04-30 의 시총 500 위 순위의 회사 저장

            now_state_3d, _ = self.encoder_12m(random_date + MonthEnd(1), companies_by_R_now) # 2011-05-31 ~ 2012-04-30 (500,12,22) shape 3d tensor return
            next_state_3d, _ = self.encoder_12m(next_date + MonthEnd(1), companies_by_R_now) # 2011-06-31 ~ 2012-05-31

            # replay buffer 채워질때 까지 채우는 코드
            ##우리식 대로 바꾸기
            key_table_ = self.company_table_per_month[next_date]

            action, codes, idxs = self.actor(now_state_3d, key_table=key_table_)

            reward = action.clone().detach().numpy().reshape(-1).T @ \
                     self.reward_list[(self.reward_list['code'].isin(codes)) & (self.reward_list['mdate'] == random_date)][
                         'ret_1m']
            # observation_, reward, done, info = env.step(action)

            # print(random_date)
            self.remember(now_state_3d.detach().numpy(), action, reward, next_state_3d.detach().numpy(), idxs, codes)
            _critic_loss, _actor_loss = self.learn(key_table=key_table_, tau = tau, gamma = gamma)

            train_critic_loss[i] = _critic_loss
            train_actor_loss[i] = _actor_loss

        # train 된 target actor parameter 저장
        torch.save(self.target_actor, save_path)

        # plotting the effect of training
        pd.Series(train_critic_loss).plot(color = 'blue')
        plt.title('train_critic_loss_'+ 'epoch'+ str(epoch) + '_' + 'gamma'+ str(gamma)+ '_' + 'tau'+ str(tau)+ '_' + 'lr'+ str(self.lr1)+ '_' + 'lr'+ str(self.lr2))
        plt.savefig('./fig/train/train_critic_loss_'+ 'epoch'+ str(epoch) + '_' + 'gamma'+ str(gamma)+ '_' + 'tau'+ str(tau)+ '_' + 'lr'+ str(self.lr1)+ '_' + 'lr'+ str(self.lr2)+'.png')
        plt.show()

        pd.Series(train_actor_loss).plot(color='blue')
        plt.title('train_actor_loss_'+ 'epoch'+ str(epoch) + '_' + 'gamma'+ str(gamma)+ '_' + 'tau'+ str(tau)+ '_' + 'lr'+ str(self.lr1)+ '_' + 'lr'+ str(self.lr2))
        plt.savefig('./fig/train/train_actor_loss_'+ 'epoch'+ str(epoch) + '_' + 'gamma'+ str(gamma)+ '_' + 'tau'+ str(tau)+ '_' + 'lr'+ str(self.lr1)+ '_' + 'lr'+ str(self.lr2)+'.png')
        plt.show()



  #  input parameters : train_begin, train_end,valid_begin_date(), valid_end_date(), load_name
    def validation(self, begin_date, end_date, epoch, gamma, tau):
        valid_returns = pd.Series()
        valid_sharpe_ratio = pd.Series()
        valid_vol = pd.Series()
        for i in self.datelist[self.datelist.index(begin_date):self.datelist.index(end_date)]: # date : 2011.01 ~ 2016.12 (check need)

            """
                i의 의미?
                예를들어 2012/3/31 이 i라면,
                2011/4/30~ 2012/3/31 데이터가 들어가서 3d state를 만든후
                2012 4월의 포트폴리오를 설계하여 수익률,샤프레이시오 를 계산하는 코드
                
                encoder 12m 구조자체가 input 으로 2012/4/30이 들어가면 2011/4/30~ 2012/3/31 의 3d tensor가 나오는 구조임
            """

            mdate_start =  self.datelist[self.datelist.index(i) - 11]

            # Slice data by date
            data_12m = self.processed_data[(self.processed_data['mdate'] >= mdate_start) & (self.processed_data['mdate'] <= i)].copy(deep=True)

            # t-11~ t시점까지 12개월 모두 존재하는 종목 코드 filtering한 것.
            data_12m = data_12m.dropna()

            all_month_exist_company = data_12m['code'].value_counts().index[data_12m['code'].value_counts().values == 12]

            #data_12m = data_12m[data_12m['code'].isin(all_month_exist_company)]

            temp = self.processed_data[self.processed_data['mdate'] == i]
            temp = temp[temp['code'].isin(all_month_exist_company)]

            temp = temp.sort_values(self.standard_column, ascending = False)
            temp = temp.iloc[:500].reset_index(drop = True)

            valid_i_code = temp['code']

            #
            valid_now_state_3d, _ = self.encoder_12m(i+MonthEnd(1), valid_i_code)

            valid_pf_weights, valid_pf_codes, _ = self.target_actor(valid_now_state_3d, key_table = valid_i_code)
            valid_pf_weights = valid_pf_weights.detach().numpy().reshape(-1)
            ############################ 종목, 비중 뽑기 완료 #######################################
            ############################ 평가 시작 #################################################
            valid_result = pd.DataFrame({'pf_weights':valid_pf_weights,'pf_codes':valid_pf_codes})
            next_month_data = self.processed_data[self.processed_data['mdate']==(i+MonthEnd(1))]

            valid_result_1 = valid_result.merge(next_month_data,left_on='pf_codes' ,right_on='code',how='left')
            i_valid_pf_return = np.sum(valid_result_1['ret_1m'].to_numpy()*valid_result_1['pf_weights'].to_numpy())
            i_valid_pf_vol = np.sqrt(np.sum((valid_result_1['ret_1m'].to_numpy()**2)*(valid_result_1['pf_weights'].to_numpy()**2)))
            i_valid_pf_sharpe_ratio = i_valid_pf_return/i_valid_pf_vol

            valid_returns[i] = i_valid_pf_return
            valid_vol[i] = i_valid_pf_vol
            valid_sharpe_ratio[i] = i_valid_pf_sharpe_ratio

            print(i)
        cumprod_returns = np.cumprod(valid_returns+1)

        # kospi_return = self.data[['vwret_kospi','mdate']]
        # kospi_return = kospi_return[(kospi_return['mdate']>=begin_date )& (kospi_return['mdate']<end_date)].set_index('mdate')
        # kospi_return = np.cumprod(1+kospi_return['vwret_kospi'])

        # kospi_return.plot()

        # make a plot of cumulative return of validation results
        cumprod_returns.plot(color = 'red')
        plt.title('validation Cumulative returns plot_'+ 'epoch'+ str(epoch) + '_' + 'gamma'+ str(gamma)+ '_' + 'tau'+ str(tau)+ '_' + 'lr'+ str(self.lr1)+ '_' + 'lr'+ str(self.lr2))
        plt.savefig('./fig/validation/validation Cumulative returns plot_'+ 'epoch'+ str(epoch) + '_' + 'gamma'+ str(gamma)+ '_' + 'tau'+ str(tau)+ '_' + 'lr'+ str(self.lr1)+ '_' + 'lr'+ str(self.lr2)+'.png')
        plt.show()

        valid_sharpe_ratio.plot(color = 'red')
        plt.title('validation Sharpe ratio plot_'+ 'epoch'+ str(epoch) + '_' + 'gamma'+ str(gamma)+ '_' + 'tau'+ str(tau)+ '_' + 'lr'+ str(self.lr1)+ '_' + 'lr'+ str(self.lr2))
        plt.savefig('./fig/validation/validation Sharpe ratio plot_'+ 'epoch'+ str(epoch) + '_' + 'gamma'+ str(gamma)+ '_' + 'tau'+ str(tau)+ '_' + 'lr'+ str(self.lr1)+ '_' + 'lr'+ str(self.lr2)+'.png')
        plt.show()
    def test(self, model, begin_date, end_date, epoch, gamma, tau):
        test_returns = pd.Series()
        test_sharpe_ratio = pd.Series()
        test_vol = pd.Series()
        for i in self.datelist[self.datelist.index(begin_date):self.datelist.index(
                end_date)]:  # date : 2011.01 ~ 2016.12 (check need)

            """
                i의 의미?
                예를들어 2012/3/31 이 i라면,
                2011/4/30~ 2012/3/31 데이터가 들어가서 3d state를 만든후
                2012 4월의 포트폴리오를 설계하여 수익률,샤프레이시오 를 계산하는 코드

                encoder 12m 구조자체가 input 으로 2012/4/30이 들어가면 2011/4/30~ 2012/3/31 의 3d tensor가 나오는 구조임
            """

            mdate_start = self.datelist[self.datelist.index(i) - 11]

            # Slice data by date
            data_12m = self.processed_data[
                (self.processed_data['mdate'] >= mdate_start) & (self.processed_data['mdate'] <= i)].copy(deep=True)

            # t-11~ t시점까지 12개월 모두 존재하는 종목 코드 filtering한 것.
            data_12m = data_12m.dropna()

            all_month_exist_company = data_12m['code'].value_counts().index[
                data_12m['code'].value_counts().values == 12]

            # data_12m = data_12m[data_12m['code'].isin(all_month_exist_company)]

            temp = self.processed_data[self.processed_data['mdate'] == i]
            temp = temp[temp['code'].isin(all_month_exist_company)]

            temp = temp.sort_values(self.standard_column, ascending=False)
            temp = temp.iloc[:500].reset_index(drop=True)

            test_i_code = temp['code']

            #
            test_now_state_3d, _ = self.encoder_12m(i + MonthEnd(1), test_i_code)

            test_pf_weights, test_pf_codes, _ = model(test_now_state_3d, key_table=test_i_code)
            test_pf_weights = test_pf_weights.detach().numpy().reshape(-1)
            ############################ 종목, 비중 뽑기 완료 #######################################
            ############################ 평가 시작 #################################################
            test_result = pd.DataFrame({'pf_weights': test_pf_weights, 'pf_codes': test_pf_codes})
            next_month_data = self.processed_data[self.processed_data['mdate'] == (i + MonthEnd(1))]

            test_result_1 = test_result.merge(next_month_data, left_on='pf_codes', right_on='code', how='left')
            i_test_pf_return = np.sum(test_result_1['ret_1m'].to_numpy() * test_result_1['pf_weights'].to_numpy())
            i_test_pf_vol = np.sqrt(
                np.sum((test_result_1['ret_1m'].to_numpy() ** 2) * (test_result_1['pf_weights'].to_numpy() ** 2)))
            i_test_pf_sharpe_ratio = i_test_pf_return / i_test_pf_vol

            test_returns[i] = i_test_pf_return
            test_vol[i] = i_test_pf_vol
            test_sharpe_ratio[i] = i_test_pf_sharpe_ratio

            print(i)
        cumprod_returns = np.cumprod(test_returns + 1)

        # kospi_return = self.data[['vwret_kospi','mdate']]
        # kospi_return = kospi_return[(kospi_return['mdate']>=begin_date )& (kospi_return['mdate']<end_date)].set_index('mdate')
        # kospi_return = np.cumprod(1+kospi_return['vwret_kospi'])

        # kospi_return.plot()

        # make a plot of cumulative return of validation results
        cumprod_returns.plot(color='red')
        plt.title('test Cumulative returns plot_' + 'epoch' + str(epoch) + '_' + 'gamma' + str(
            gamma) + '_' + 'tau' + str(tau) + '_' + 'lr' + str(self.lr1) + '_' + 'lr' + str(self.lr2))
        plt.savefig('./fig/test/validation Cumulative returns plot_' + 'epoch' + str(epoch) + '_' + 'gamma' + str(
            gamma) + '_' + 'tau' + str(tau) + '_' + 'lr' + str(self.lr1) + '_' + 'lr' + str(self.lr2) + '.png')
        plt.show()

        #test_sharpe_ratio.plot(color='blue')
        #plt.title('test Sharpe ratio plot_' + 'epoch' + str(epoch) + '_' + 'gamma' + str(gamma) + '_' + 'tau' + str(
        #        tau) + '_' + 'lr' + str(self.lr1) + '_' + 'lr' + str(self.lr2))
        #plt.savefig('./fig/test/validation Sharpe ratio plot_' + 'epoch' + str(epoch) + '_' + 'gamma' + str(
        #    gamma) + '_' + 'tau' + str(tau) + '_' + 'lr' + str(self.lr1) + '_' + 'lr' + str(self.lr2) + '.png')
        #plt.show()