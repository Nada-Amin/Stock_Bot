from datetime import date
from oop_final import StockBot
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle  #to load a saved model
import base64  #to open .gif files in streamlit app
# import adam optimizer
from keras.optimizers import Adam
def predict(RNN_params):
    x = StockBot(**RNN_params)
    #print(x.company)
    #print(x.start_date)
    #print(x.end_date)
    # save the stock data
    df = x.save_stock(RNN_params)
    X_train, y_train, X_test, y_test, scaler = x.preprocess(df, RNN_params)
    # build the model
    model = x.build_model(RNN_params,X_train)
    #model.summary()
    history, model = x.train_model(model, X_train, y_train, RNN_params)
    y_pred_train, y_train, y_pred_test, y_test = x.test_model(model, X_train,y_train,X_test, y_test, scaler)
    x.evaluation(y_test, y_pred_test)
    # flatten the y_train_pred and y_test_pred
    y_pred_train = y_pred_train.flatten()
    y_pred_test = y_pred_test.flatten()
    return df,model,RNN_params,scaler,y_pred_train,y_pred_test,X_test,y_test,x
    # plot the predictions
    # x.predict_next_days(df,model,RNN_params,scaler,y_pred_train,y_pred_test,X_test,y_test)

    # # save the model
    # x.save_model(model,scaler,RNN_params)
def show_res(df,model,RNN_params,scaler,y_pred_train,y_pred_test,X_test,y_test,x):
    future=x.predict_next_days(df,model,RNN_params,scaler,y_pred_train,y_pred_test,X_test,y_test)
    return future
@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page:',['Home','Prediction']) #two pages

if app_mode=='Home':
    st.title('Stock Prediction :')  
    st.image('stock-trading-bot.jpg')
    st.subheader("Description :")
    st.markdown("This is our stock bot dedicated for scraping data from different companies like TSLA, TWTR and GM due to the selected end date using 4 models: LSTM, LSTMsimp, GRU, RNN.")
    # st.markdown('Dataset :')
    # data=pd.read_csv('TSLA.csv')
    # st.write(data.head())
    # st.bar_chart(data[['Adj Close','Open']].head(20))
elif app_mode == 'Prediction':
    # st.sidebar.header("Choose the data :")
    company=st.sidebar.text_input('Enter Company Sympol:')
    model_name=st.radio('Choose Model',['LSTM', 'LSTMsimp', 'GRU', 'RNN'])
    # split_date=st.sidebar.date_input('split_date')
    # start_date=st.sidebar.date_input('start_date') 
    end_date=st.sidebar.date_input('End Date:')
    RNN_params = {"company": company,
                "window": 120,
                "model_name": model_name,
                "units": 128,
                "layers": 4,
                "dropout": 0.5,
                "loss": 'mean_squared_error',
                "optim": Adam,
                'optimizer':'adam',
                "lr": 0.0001,
                "epochs": 1,
                'FC_size': 32,
                "batch_size": 32,
                "earlystop": {"patience": 10,"min_delta": 0.00001},
                "reduce_lr": {"factor": 0.1,
                                "patience": 5,
                                "min_delta": 0.0001},
                'split_date': '2021/01/01',
                'start_date': '2010-01-01',
                'end_date': end_date,
                'days': 30
                }
# for k, v in RNN_params().items():
#             print(f'{k=}, {v=}')
    # dict=str(RNN_params)
    # st.header(dict)
    # feature_list=[]
    # single_sample = np.array(feature_list).reshape(1,-1)
    if st.button("Predict"):
            RNN_params['split_date'] = str(date(int(end_date.year)-1, end_date.month, end_date.day))
            df,model,RNN_params,scaler,y_pred_train,y_pred_test,X_test,y_test,x=predict(RNN_params)
            # container=st.container()
            # container.show_res(df,model,RNN_params,scaler,y_pred_train,y_pred_test,X_test,y_test,x)
            future=show_res(df,model,RNN_params,scaler,y_pred_train,y_pred_test,X_test,y_test,x)
            st.write(future)
            # chart_data = pd.DataFrame(
            # future['Adj Close','train_test_pred'])
            #st.line_chart(data=future,y=['Adj Close'[:-30],'Adj Close'[-30:],'train_test_pred'])
            fig, ax = plt.subplots(figsize=(16,9))
            ax.plot(future.index[:-RNN_params['days']], future['Adj Close'][:-RNN_params['days']], color='b')
            ax.plot(future.index[-RNN_params['days']:], future['Adj Close'][-RNN_params['days']:], color='r')
            # plot the entire predicted test set in green
            ax.plot(future.index, future['train_test_pred'], color='g')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Adj Close')
            ax.legend(['Real', 'Forecast', 'Train test'])
            st.pyplot(fig)
        