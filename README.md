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
