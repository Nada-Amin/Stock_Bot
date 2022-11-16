# %% [markdown]
# # Import Packages

import datetime as dt
import os
import pickle
# suppress warnings
import warnings

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import pandas
import pandas as pd
import pandas_datareader.data as web
from keras import backend as K
# import early stopping
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import GRU, LSTM, Dense, Dropout, SimpleRNN
# %%
# import the rnn and bilstm models
from keras.models import Sequential, load_model
# import adam optimizer
from keras.optimizers import Adam
from matplotlib import style
#import pandas.plotting
from pandas.plotting import register_matplotlib_converters
#import pandas.testing
from pandas.testing import assert_frame_equal
# import the metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
# import the scaler
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

from math import sqrt

# calculate the MDA
# calculate the MAPE
# calculate the MAE
# calculate the RMSE
from sklearn.metrics import (max_error, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)

style.use('ggplot')

class StockBot:
    def __init__(self,
                company,
                window,
                model_name,
                units,
                layers,
                dropout,
                loss,
                optim,
                optimizer,
                lr,
                epochs,
                FC_size,
                batch_size,
                earlystop,
                reduce_lr,
                split_date,
                start_date,
                end_date,
                days):
                self.company = company
                self.window = window
                self.model_name = model_name
                self.units = units
                self.layers = layers
                self.dropout = dropout
                self.loss = loss
                self.optim = optim
                self.optimizer = optimizer
                self.lr = lr
                self.epochs = epochs
                self.FC_size = FC_size
                self.batch_size = batch_size
                self.earlystop = earlystop
                self.reduce_lr = reduce_lr
                self.split_date = split_date
                self.start_date = start_date
                self.end_date = end_date
                self.days = days

    def save_stock(self,RNN_params):
        df = web.DataReader(RNN_params['company'], 'yahoo', RNN_params['start_date'], RNN_params['end_date'])
        if not os.path.exists('stock_data'):
            os.makedirs('stock_data')
        df.to_csv('stock_data/{}.csv'.format(RNN_params['company']))
        return df
        
    def preprocess(self,df,RNN_params):
        # scale the data
        
        # drop everything except the Adj Close as it is the only feature
        df = df.drop(['Open','High', 'Low', 'Close', 'Volume'], axis=1)

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df)
        train = df[:RNN_params['split_date']]
        test = df[RNN_params['split_date']:]
        print('train shape: ', train.shape)
        print('test shape: ', test.shape)
        # save the size of the training set
        train_size = len(train)
        
        # split the scaled data into training and testing based on the train_size
        train = scaled_data[:train_size,:]
        test = scaled_data[train_size:,:]
        print('train shape: ', train.shape)
        print('test shape: ', test.shape)
        # plot the scaled data
        plt.figure(figsize=(16,8))
        plt.title('Scaled Data')
        plt.xlabel('Date')
        plt.ylabel('Scaled Adj Close Price USD ($)')
        plt.plot(scaled_data, label='Scaled Data')
        plt.legend()
        plt.show()
        # create the X_train and y_train
        X_train = []
        y_train = []
        for i in range(RNN_params['window'], len(train)):
            X_train.append(train[i-RNN_params['window']:i, 0])
            y_train.append(train[i, 0])
        # convert to numpy arrays
        X_train, y_train = np.array(X_train), np.array(y_train)
        # reshape the data
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        

        # create the X_test and y_test
        X_test = []
        y_test = []
        for i in range(RNN_params['window'], len(test)):
            X_test.append(test[i-RNN_params['window']:i, 0])
            y_test.append(test[i, 0])
        # convert to numpy arrays
        X_test, y_test = np.array(X_test), np.array(y_test)
        # reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, y_train, X_test, y_test, scaler

    def build_model(self,RNN_params,X_train):
        # build the model
        model = Sequential()
        if RNN_params['model_name'] == 'LSTM':
            for i in range(RNN_params['layers']-1):
                if i == 0:
                    model.add(LSTM(units=RNN_params['units'], return_sequences=True, input_shape=(X_train.shape[1], 1)))
                else:
                    model.add(LSTM(units=RNN_params['units'], return_sequences=True))
                model.add(Dropout(RNN_params['dropout']))
            
            model.add(LSTM(units=RNN_params['units']))
            model.add(Dropout(RNN_params['dropout']))
            model.add(Dense(units=1))
            model.compile(optimizer=RNN_params['optimizer'], loss=RNN_params['loss'])
            
        elif RNN_params['model_name'] == 'LSTMsimp':
            for i in range(RNN_params['layers']-1):
                if i == 0:
                    model.add(LSTM(units=RNN_params['units'], return_sequences=True, input_shape=(X_train.shape[1], 1)))
                else:
                    model.add(LSTM(units=RNN_params['units'], return_sequences=True))
                model.add(Dropout(RNN_params['dropout']))
            model.add(LSTM(units=RNN_params['units']))
            model.add(Dropout(RNN_params['dropout']))
            model.add(Dense(RNN_params['FC_size'], activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer=RNN_params['optimizer'], loss=RNN_params['loss'])

        elif RNN_params['model_name'] == 'GRU':
            for i in range(RNN_params['layers']-1):
                if i == 0:
                    model.add(GRU(units=RNN_params['units'], return_sequences=True, input_shape=(X_train.shape[1], 1)))
                else:
                    model.add(GRU(units=RNN_params['units'], return_sequences=True))
                model.add(Dropout(RNN_params['dropout']))
            model.add(GRU(units=RNN_params['units']))
            model.add(Dropout(RNN_params['dropout']))
            model.add(Dense(units=1))
            model.compile(optimizer=RNN_params['optimizer'], loss=RNN_params['loss'])

        elif RNN_params['model_name'] == 'RNN':
            for i in range(RNN_params['layers']):
                if i == 0:
                    model.add(SimpleRNN(units=RNN_params['units'], return_sequences=True, input_shape=(X_train.shape[1], 1)))
                else:
                    model.add(SimpleRNN(units=RNN_params['units'], return_sequences=True))
                model.add(Dropout(RNN_params['dropout']))
            model.add(SimpleRNN(units=RNN_params['units']))
            model.add(Dropout(RNN_params['dropout']))
            model.add(Dense(units=1))
            model.compile(optimizer=RNN_params['optimizer'], loss=RNN_params['loss'])
        
        else:
            print('Please choose a valid model')
        return model

        # train the model
    def train_model(self, model, X_train, y_train, RNN_params):
        # train the model and add patience with min_delta to stop training when the loss is not improving,
        # also reduce the learning rate with factor from RNN_params and patience and min delta
        # to stop reducing the learning rate when the loss is not improving, the loss is from RNN_params mean_squared_error
        # the optimizer is from RNN_params
        # the metrics is from RNN_params
        # the batch size is from RNN_params
        # the epochs is from RNN_params
        # the RNN_params in the shape "earlystop": {"patience": 20,"min_delta": 0.00001},
                #"reduce_lr": {"factor": 0.1,
                            #  "patience": 10,
                            # "min_delta": 0.0001}
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=RNN_params["reduce_lr"]["factor"], patience=RNN_params["reduce_lr"]["patience"], min_delta=RNN_params["reduce_lr"]["min_delta"], mode='min')
        earlystop = EarlyStopping(monitor='loss', min_delta=RNN_params["earlystop"]["min_delta"], patience=RNN_params["earlystop"]["patience"], mode='min', verbose=1)
        history = model.fit(X_train, y_train, epochs=RNN_params['epochs'], batch_size=RNN_params['batch_size'], callbacks=[reduce_lr, earlystop])
        return history, model

    # plot the training loss and validation loss, note that model has no attribute history
    # so we need to use the history variable to plot the loss
    def plot_loss(self,history):
        plt.figure(figsize=(16,8))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()


    # test the model
    def test_model(self, model, X_train,y_train,X_test, y_test, scaler):
        # predict the test set
        y_pred_train = model.predict(X_train)
        # get the data of the scaler
        #print(scaler.data_max_)
        # inverse the scaling
        y_pred_train = scaler.inverse_transform(y_pred_train)
        # inverse the y_test but reshape it to the input shape of the scaler
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))


        # predict the test set
        y_pred_test = model.predict(X_test)
        # get the data of the scaler
        #print(scaler.data_max_)
        # inverse the scaling
        y_pred_test = scaler.inverse_transform(y_pred_test)
        # inverse the y_test but reshape it to the input shape of the scaler
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        
        return y_pred_train, y_train, y_pred_test, y_test

    # get the last window and predict the next range of days
    def predict_next_days(self, df,model,RNN_params,scaler,y_pred_train,y_pred_test,X_test,y_test):
        days = RNN_params['days']

        last_window = X_test[-1]
        #print(last_window.shape)
        last_window = np.reshape(last_window, (1, last_window.shape[0], 1))
        #print(last_window.shape)
        #print(last_window[0])

        predictions = []
        for i in range(days):
            pred = model.predict(last_window)
            predictions.append(pred[0][0])
            last_window = np.append(last_window[0][1:], pred)
            last_window = np.reshape(last_window, (1, last_window.shape[0], 1))
        #print(predictions)
        #print(predictions.shape)

        # inverse the predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
        #print(predictions)

        # plot the predictions
        # plt.figure(figsize=(16,8))
        # plt.plot(predictions, color='blue', label='Predicted Stock Price')
        # plt.title('Stock Price Prediction')
        # plt.xlabel('Time')
        # plt.ylabel('Stock Price')
        # plt.legend()


        # copy the dataframe to a new dataframe called future
        future = df.copy()
        #print(future.tail(10))
        #print(future.shape)
        # add 30 rows to the dataframe
        # calculate how many rows to add with the given days taking into account the holidays saterday and sunday
        rows = days + (days//5)*2
        #print(rows)
        for i in range(rows):
            future.loc[future.index.max() + pd.DateOffset(days=1)] = [np.nan] * len(future.columns)

        #print(future.tail(10))
        # reset the index
        future.reset_index(inplace=True)
        #print(future.tail(10))
        # get the name of the day of the date column 
        future['day'] = future['Date'].dt.day_name()
        #print(future.tail(10))
        # drop the row that has satensday and sunday
        future = future[future['day'] != 'Saturday']
        future = future[future['day'] != 'Sunday']
        # drop the day column
        future.drop('day', axis=1, inplace=True)

        # set the date column as the index
        future.set_index('Date', inplace=True)
        #print(future.tail(31))
        # strip the predictions list 
        predictions = predictions.flatten()
        #print(predictions)
        # add the predictions to the last 'days' rows of the dataframe
        future['Adj Close'].iloc[-days:] = predictions


        #print(future.tail(31))
        # add the X_train_pred and y_train_pred to the dataframe under one column called train_test_pred
        future['train_test_pred'] = np.nan

        # train_test_pred = the RNN['window'] of future['Adj Close'] 
        train_test_pred = future['Adj Close'].iloc[:RNN_params['window']]


        # print the values of train_test_pred
        #print(train_test_pred.values)
        # convert the train_test_pred values to a numpy array
        train_test_pred = np.array(train_test_pred)
        # concatenate the train_test_pred and y_pred_train
        train_test_pred = np.concatenate((train_test_pred, y_pred_train))

        # get another RNN['window'] of future['Adj Close'] after the y_pred_train
        to_be_added = future['Adj Close'].iloc[RNN_params['window']+len(y_pred_train):RNN_params['window']+len(y_pred_train)+RNN_params['window']]
        # convert the to_be_added values to a numpy array
        to_be_added = np.array(to_be_added)
        # concatenate the train_test_pred and to_be_added
        train_test_pred = np.concatenate((train_test_pred, to_be_added))

        # concatenate the train_test_pred and y_pred_test
        train_test_pred = np.concatenate((train_test_pred, y_pred_test))
        # add the train_test_pred to the future dataframe - days
        future['train_test_pred'].iloc[:-days] = train_test_pred



        # # copy the RNN_params['window'] rows from the future['Adj Close'] to the future['train_test_pred']
        # future['train_test_pred'].iloc[:RNN_params['window']] = future['Adj Close'].iloc[:RNN_params['window']]
        # # copy the y_pred_train after the RNN_params['window'] to the future['train_test_pred']
        # future['train_test_pred'].iloc[RNN_params['window']:RNN_params['window']+len(y_pred_train)] = y_pred_train
        # # copy the y_pred_test after the RNN_params['window']+len(y_pred_train) to the future['train_test_pred']
        # slicer = 2*(RNN_params['window'])+len(y_pred_train)
        # future['train_test_pred'].iloc[RNN_params['window']+len(y_pred_train):slicer] = future['Adj Close'].iloc[RNN_params['window']+len(y_pred_train):slicer]
        # # copy the y_pred_test after the 2*RNN_params['window']+len(y_pred_train) to the future['train_test_pred']
        # future['train_test_pred'].iloc[slicer:] = y_pred_test





        #print(y_pred_train.shape)
        #print(y_pred_train[0])

        # plot the Adj Close price in blue without the last 'days' rows and in red with the last 'days' rows
        # plt.figure(figsize=(16,8))
        # plt.plot(future['Adj Close'][:-days], color='blue', label='Real Stock Price')
        # plt.plot(future['Adj Close'][-days:], color='red', label='forecast Stock Price')
        # plt.plot(future['train_test_pred'], color='green', label='Train Test Predictions')
        # plt.title('Stock Price Prediction')
        # plt.xlabel('Time')
        # plt.ylabel('Stock Price')
        # plt.legend()
        return future
    
    
    def evaluation(slef, real_stock_price, predicted_stock_price):
        rmse = sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        print('Test RMSE: %.3f' % rmse)



        mae = mean_absolute_error(real_stock_price, predicted_stock_price)
        print('Test MAE: %.3f' % mae)


        mape = mean_absolute_percentage_error(real_stock_price, predicted_stock_price)
        print('Test MAPE: %.3f' % mape)


        mda = max_error(real_stock_price, predicted_stock_price)
        print('Test MDA: %.3f' % mda)
    # save the model
    def save_model(slef, model, scaler, RNN_params):
        # save the model with the name of the company from the RNN_params dictionary and name of the model, save in a folder called models and inside this folder create a folder with the name of the company
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('models/'+RNN_params['company']):
            os.makedirs('models/'+RNN_params['company'])
        model.save('models/'+RNN_params['company']+'/'+RNN_params['model_name']+'.h5')
        # save the scaler with the name of the company from the RNN_params dictionary and name of the model
        pickle.dump(scaler, open('models/'+RNN_params['company']+'/'+RNN_params['model_name']+'_scaler.pkl', 'wb'))