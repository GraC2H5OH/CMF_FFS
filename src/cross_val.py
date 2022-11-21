import sys
sys.path.append(r"E:\CMF\CMF_FFS\src/")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from metrics_calculation import metrics
from preprocess import preprocessing

def timeseriesCVscore(data, model):
    
    y_list = []
    pred_list = []
    
    data, y = preprocessing(data)
    

    values = data

    tscv = TimeSeriesSplit(n_splits=3, test_size=4) 

    for train, test in tscv.split(values):
        
        scaler = StandardScaler()
        scaler.fit(values.iloc[train])
        
        train_scaled = scaler.transform(values.iloc[train])

        model.fit(train_scaled, y.iloc[train])
        pred = model.predict(values.iloc[test])
        
        y_list.append(y.iloc[test])
        pred_list.append(pred)
    
    
    mape, wape, mse = metrics(y_list, pred_list, one_model=False)
    
    # There is some troubles with wape, sometimes wape equals to nan
    
    return  mape, np.nan_to_num(wape), mse
