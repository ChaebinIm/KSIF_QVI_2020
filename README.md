# KSIF_QVI_2020
KSIF_QVI_2020 Python Source

### About this Repository
###### There are 4 projects in this repository. These are the source codes I contributed when I was in KSIF(KAIST Student Institute Fund) QVI(Quantitative Value Investment) Team.
###### There are the list of source codes
1. Alpha Portfolio
2. DDPG(Deep Deterministic Policy Gradient)
3. Distress detector
4. FML(Financial Machine Leanring)

###### 1 and 2 are the algorithm with RL(Reinforcement Learning), 3 is one of an algorithm of machine learning(with logistic regression), 4 is an algorithm with Deep Learning.
###### I want to introduce Alpha Portfolio in this README because I almost concentrate on it when I was in KSIF QVI Team.
###### If I have a free time, I will additionally introduce the 2, 3, and 4. But, maybe it is quite hard because it can be so long and has no impact.

--------------------------------------------------------------
### About Alpha Portfolio
###### This algorithm is made with RL(Reinforcement Learning), in detail, which is about the logic called policy gradient
###### It is based on a paper, called "Alphaportfolio for Investment and Economically intepretable AI".
###### The url of this paper is <https://faculty.comm.virginia.edu/sdb7e/files/mcintireSeminars/cong_AlphaPortfolio.pdf>
###### In the network of the model, Transformer is used. Transformer is made by Deep Mind, Google. It is so famous in the field natural language processing, using parallel self-attention algorithm.
###### I'm supposing that you know about Attention, but if you don't, you have to study about it.
###### Additionally, If you know more about Transformer, you can read the paper called "Attention is All You Need".


###### Now, I want to talk about the detail algorithm of Alpha portfolio.
#### Structure of Transformer network.

<img src="https://user-images.githubusercontent.com/44806420/128182382-c51faadf-64a0-40e4-b9c0-687f147c0f02.jpg"  width="380" height="500">

###### Transformer network don't use RNN networks, but use just attention network.
###### Transformer is made of two parts, called encoder and decoder(like Seq2Seq models)
###### With encoder, it vectories the data, and with decoder, it makes the output(in machine translation, it can be other language).
###### With Transformer, we use the encoder on its own, but in the case of decoder, we change it into CAAN(Cross-asset Attention Network) network.

<img src= "https://user-images.githubusercontent.com/44806420/128356572-ef077a84-3431-4cf3-bc0e-822f811d7b4a.png" width="700" height="300">

###### Yeah, in other words, Alpha Portfolio is made of the two networks, Transformer encoder & CAAN. Through it, you can process the data and get the stocks and ratio of your portfolio. Now, I want to talk about How the networks, TE(Transformer Network) and CAAN(Cross-Asset Attention network) is used in this model.


#### Transformer Encoder(TE) and CAAN(Cross-Asset Attention Network)
###### TE network is used to take the correlation among times into account. The multi-head attention network is used in the range of period. So, with TE, we could consider the correlation between past and future. For detail, I have to talk about the data shape. The input of this data is 3-d data. Each demension is made with (Period, stocks, fundamentals). With this structure, if you put it into TE, the model apply the time-changing of the data.
###### CAAN is pretty similar with TE. But, you should reshape the data to take the correlation among stocks into account. The data shape is (Stocks, period, fundamentals). With it, the model apply the correlation among stocks of the data. We did it with the python, pytorch library because we could easily find the source code of Transformer on the internet. We refered the source code and just changed it into what we want to use. The final part of CAAN, before putting it into feed forward network, you have to reshape the data into 2-d shape to put it into the deep neural network. So, at that time, you use the data shape (stocks, period * features).
###### After the neural network, you can get shape (stocks, 1), meaning the "winner score". With it, you can get the stock's ratio in your portfolio. Then, How this model can be trained?
###### It's simple. With the winner score, you can make a portfolio and you can caluculate your portfolio return. After some repetition, you can get the series of portfolio return, so you can calculate the sharpe ratio of the portfolio in the series.
###### Here, the sharpe ratio is the "reward" in reinforcement learning. Yeah, it can be hard to understand. Now, I want to talk about the reinforcement learning application into this algorithm

#### Reinforcement Learning
###### In this model, the reinforcement learning algorithm, policy gradient, is used. simply introducing, it is just updating the policy to get the optimal return(cumulative sum of rewards). Here is the list of application of policy gradient algorithm in this model.
- Environment : Historical data
- State : market data during randomly selected period
- Policy : TE & CAAN network
- Reward : Shapre ratio of the portfolio in the period(state)

###### The Reinforcement learning algorithm training process is really simple.
1. By environment, get a state.
2. With the state, the policy is processed. In the process, TE & CAAN network is used. In my model, the period length is 12month(monthly data). So, the process must be used 12 times.
3. After 12 times(months), you can get the reward(sharpe ratio of the model's portfolio).
4. The model is trained to maximize the reward(sharpe ratio).

###### With the process above, you can construct the model infrastructure. It's the end. Really simple.
