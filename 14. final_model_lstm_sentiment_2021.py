
import numpy as np
import time
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from ExTrajectory_nooption_lstm import ExpertTraj
# plt.ion()
from torch.utils.data import DataLoader, Sampler
# from torchvision import datasets,transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import sklearn.metrics as metrics

def sample(train_state, train_action, batch_size):
    # print('self.n_transitions',self.n_transitions)
    n_transitions = train_state.shape[0]
    indexes = np.random.randint(0, n_transitions, size=batch_size)
    state, action = [], []
    # print('self.train_state', self.train_state.shape)
    for i in indexes:
        s = train_state[i]
        a = train_action[i]
        # o = self.train_initial_option[i]
        state.append(np.array(s, copy=False))
        action.append(np.array(a, copy=False))

    return np.array(state), np.array(action)
model = 'MSFT'
d1_price = pd.read_csv(model+'.csv')
d1_score = pd.read_csv('pre_'+model+'_scores.csv')

d1_price['date'] = d1_price['date'].str[:10]
d1_price['date_year'] = d1_price['date'].str[:4]

d1_price = d1_price[d1_price['date_year'] == '2021']

d1_price.rename(columns = {'date':'release_date'}, inplace = True)

d1_all = pd.merge(d1_price,d1_score,on='release_date')
print(d1_all.head(10))
# df1 = df.copy()
# print(df1.head(10))
# d1_all['date'] = d1_all.date.str[:4]
# print(df1.head(10))
# df = d1_all[d1_all['date_year']=='2019']
df1=d1_all[['close','sentiment_score']]


scaler1=MinMaxScaler(feature_range=(0,1))
df1['close']=scaler1.fit_transform(np.array(df1['close']).reshape(-1,1))

# print(df1['close'])
df1 = df1.values
# scaler2=MinMaxScaler(feature_range=(0,1))
# df1['sentiment_score']=scaler2.fit_transform(np.array(df1['sentiment_score']).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.80)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:]
print('test',test_data.shape)
# print(train_data.shape)
import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), :]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
# X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 2)
# X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)





# print(X_train.shape)
# print(y_train.shape)
# exit()
class Policy_LSTM(nn.Module):
    def __init__(self, state_dim, hidden_layers=64):
        super(Policy_LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(state_dim, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)
        torch.nn.init.xavier_normal(self.linear.weight)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        # torch.nn.init.xavier_uniform(self.lstm1._all_weights)
        # torch.nn.init.xavier_uniform(self.lstm2._all_weights)

    def forward(self, input_all):
        outputs, num_samples = [], input_all.shape[0]
        h_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)

        for time_step in range(input_all.shape[1]):
            # print('input_all',input_all.shape)

            input_t = input_all[:,time_step,:]
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))  # initial hidden and cell states

            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))  # new hidden and cell states
            output = self.linear(h_t2)  # output from the last FC layer
            outputs.append(output)

        # for i in range(1):
        #     # this only generates future predictions if we pass in future_preds>0
        #     # mirrors the code above, using last output/prediction as input
        #     h_t, c_t = self.lstm1(output, (h_t, c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        #     output = self.linear(h_t2)
        #     outputs.append(output)
        # transform list to tensor
        # outputs = torch.cat(outputs, dim=1)
        return output

state_dim = 2
Rep_hidden_dim = 10
emb_dim = 1
beta = (0.5, 0.9)
resplstm = Policy_LSTM(state_dim, 56)
optim_resplstm = torch.optim.Adam(resplstm.parameters(), lr=0.0005)
loss_emb = nn.MSELoss()
loss_fn = nn.BCELoss()


for i in range(500):

    # print(np.sum(train_state,axis=0))
    x_train_, y_train_ = sample(X_train, y_train, 64)
    state = torch.FloatTensor(x_train_)
    # state = torch.unsqueeze(state, 2)
    action = torch.FloatTensor(y_train_)
    # train_action = torch.squeeze(train_action, 0)
    # print('state', state[:2])

    agent_action = resplstm(state)
    # print('agent_action',agent_action.shape)
    optim_resplstm.zero_grad()
    # agent_action= agent_action.view(-1,)
    # print('action',action.shape)
    agent_action = agent_action.view(-1, )
    # print(agent_action[:10], action[:10])
    emb_loss = loss_emb(agent_action,action)
    emb_loss.backward(retain_graph=True)
    # print('emb_loss', emb_loss)
    optim_resplstm.step()

    # plt.plot(agent_action.data.numpy())
    # plt.plot(action.data.numpy())
    # plt.show()
    #

    if i %10==0:
        with torch.no_grad():
            eval_state = X_train
            eval_action = y_train
            state = torch.FloatTensor(eval_state)
            action = torch.FloatTensor(eval_action)
            agent_action = resplstm(state)

            eval_state = X_test
            eval_action = y_test
            # print(y_test)
            state = torch.FloatTensor(eval_state)
            action = torch.FloatTensor(eval_action)
            agent_action_test = resplstm(state)

            agent_action = agent_action.data.numpy()
            agent_action_test = agent_action_test.data.numpy()

            # train_predict = scaler.inverse_transform(agent_action)
            test_predict = scaler1.inverse_transform(agent_action_test)
            action_test = test_predict
            test_predict = scaler1.inverse_transform(y_test.reshape(-1, 1))
            mae = metrics.mean_absolute_error(action_test, test_predict)
            mse = metrics.mean_squared_error(action_test, test_predict)
            rmse = np.sqrt(mse)  # or mse**(0.5)
            r2 = metrics.r2_score(action_test, test_predict)
            mape = metrics.mean_absolute_percentage_error(action_test, test_predict)

            print("Results of sklearn.metrics:")
            print("MAE:", mae)
            print("MSE:", mse)
            print("RMSE:", rmse)
            print("R-Squared:", r2)
            print("mape:", mape)


            # eval_state = X_test
            # eval_action = y_test
            # state = torch.FloatTensor(eval_state)
            # action = torch.FloatTensor(eval_action)
            # print(state.shape)
            # agent_action_test = resplstm(state)
            #
            #
            # emb_loss = loss_emb(action, agent_action)
            # print('val_emb_loss', emb_loss)
            #
            # agent_action = agent_action.data.numpy()
            # agent_action_test = agent_action_test.data.numpy()
            #
            # train_predict = scaler1.inverse_transform(agent_action)
            # test_predict = scaler1.inverse_transform(agent_action_test)

            # look_back = time_step
            # trainPredictPlot = numpy.empty_like(df1[:,0].reshape(-1,1))
            # trainPredictPlot[:, :] = np.nan
            # trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
            # # shift test predictions for plotting
            # testPredictPlot = numpy.empty_like(df1)
            # testPredictPlot[:, :] = numpy.nan
            # testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict
            # # plot baseline and predictions
            # plt.plot(scaler1.inverse_transform(df1[:,0].reshape(-1,1)))
            # plt.plot(trainPredictPlot[:,0],label='train predict')
            # plt.plot(testPredictPlot[:,0],label='test predict')
            # plt.legend()
            # plt.ylabel('Close Price')
            # plt.title(model+' 10 days for prediction')
            # plt.show()

            # ## evaluate
            #
            # state = torch.FloatTensor(eval_state)
            #
            # agent_action = resplstm(state)
            # agent_action = agent_action.data.numpy()
            # agent_action = np.reshape(agent_action,(-1,))
            # eval_action = np.reshape(eval_action,(-1,))
            # plt.plot(eval_action,label = 'Subscription rate',alpha = 0.4,color = 'r')
            # plt.plot(agent_action,label = 'Usage Rate',alpha = 0.4,color = 'b')
            # plt.legend()
            # plt.show()
            # time.sleep(5)
            # plt.close('all')
            # plt.close()
