import tensorflow as tf
print(tf.__version__)
import streamlit as st
st.write(tf.__version__)
st.write("hello world")




import streamlit as st
import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler
from jugaad_data.nse import stock_df
from streamlit_extras.no_default_selectbox import selectbox

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM,Dense
# from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import  LSTM,Dense
from tensorflow.python.keras.callbacks import EarlyStopping


all_stocks = pd.read_csv("EQUITY_L.csv")
all_stocks.columns = ['SYMBOL', 'NAME OF COMPANY', 'SERIES', 'DATE OF LISTING',
       'PAID UP VALUE', 'MARKET LOT', 'ISIN NUMBER', 'FACE VALUE']
t_date = datetime.date.today()


inp_stock = selectbox("Select Stock",all_stocks["NAME OF COMPANY"],index=0)
st.write(inp_stock)


def stock_info(in_st):

    res  = all_stocks[all_stocks["NAME OF COMPANY"]==in_st]

    li_date = datetime.datetime.strptime(res["DATE OF LISTING"].values[0],"%d-%b-%Y")

    st.table(res[["SYMBOL","NAME OF COMPANY","SERIES","DATE OF LISTING"]])

    st_symb = st.text_input("Stock Symbol",value=res["SYMBOL"][0],disabled=True)

    from_date = st.date_input("From Date",li_date + datetime.timedelta(days=1),li_date + datetime.timedelta(days=1),t_date - datetime.timedelta(days=1))

    to_date = st.date_input("To Date",t_date - datetime.timedelta(days=1),t_date - datetime.timedelta(days=1),t_date - datetime.timedelta(days=1))


    dt_sclr = MinMaxScaler((0,1))
    res_sclr = MinMaxScaler((0,1))

    nn = Sequential()
    nn.add(LSTM(50,return_sequences=True,input_shape=(4,1)))
    nn.add(LSTM(50,return_sequences=True))
    nn.add(LSTM(50))
    nn.add(Dense(1))
    nn.compile(loss='mean_squared_error',optimizer='adam')

    es = EarlyStopping(monitor='val_loss',verbose=1,patience=20)

    def get_data(symb,f_dt,t_dt):
        data = stock_df(symbol=symb,from_date=f_dt,to_date=t_dt)
        data = data['CLOSE']
        prepare_data(data)
        return data


    def prepare_data(dt):
        a, b, c, d, r = [], [], [], [], []

        for i in range(4,len(dt),1):
            a.append(dt[i-4])
            b.append(dt[i-3])
            c.append(dt[i-2])
            d.append(dt[i-1])
            r.append(dt[i])
        df = pd.DataFrame([a,b,c,d,r]).T
        df.columns = ["1","2","3","4","Result"]

        # prepro_df(df)
        return df


    def prepro_df(df):
        dt_sclr.fit(df[["1","2","3","4"]])
        res_sclr.fit(df["Result"])

        scaled_dt = dt_sclr.transform(df[["1","2","3","4"]])
        scaled_dt = pd.DataFrame(scaled_dt,columns=["1","2","3","4"])

        res_dt = res_sclr.transform(df["Result"])
        res_dt = pd.DataFrame(res_dt,columns=["Result"])

        # split_data(scaled_dt,res_dt)
        return scaled_dt, res_dt



    def split_data(x,y):
        tr_size = int(len(x)*0.85)
        ts_size = len(x)-tr_size

        tr_setX = x[:tr_size]
        tr_setY = y[:ts_size]

        ts_setX = x[ts_size:len(x)]
        ts_setY = y[ts_size:len(x)]

        # train_model(tr_setX,tr_setY,ts_setX,ts_setY)
        return tr_setX,tr_setY,ts_setX,ts_setY

    def train_model(xtrain,ytrain,xtest,ytest):
        nn.fit(xtrain,ytrain,validation_data=(xtest,ytest),callbacks=[es],epochs=150)
        last = ytest.tail(4).values
        last = pd.DataFrame(last).T
        # cal_pred(nn,last)
        return last

    def cal_pred(elem):
        pred = nn.predict(elem)
        return pred

    def convert_pred(val):
        final = res_sclr.inverse_transform(val)
        return final

    dt = get_data(st_symb,from_date,to_date)

    pro_data = prepro_df(dt)

    sc_dt, res_dt = prepro_df(pro_data)

    xtr,ytr,xts,yts = split_data(sc_dt,res_dt)

    ls_data = train_model(xtr,ytr,xts,yts)

    pred = cal_pred(ls_data)

    final_pred = convert_pred(pred)


    st.write("Prediction")
    st.write(final_pred)



       
if st.button("Get Info"):
       stock_info(inp_stock)
