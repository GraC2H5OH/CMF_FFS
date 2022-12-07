import sys
sys.path.append(r"E:\CMF\CMF_FFS\src/")

import numpy as np
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from metrics_calculation import metrics
from preprocess import preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def timeseriesCVscore(data, model, scale=True):
    
    y_list = []
    pred_list = []
    
    data, y = preprocessing(data)
    

    values = data

    tscv = TimeSeriesSplit(n_splits=3, test_size=4) 

    for train, test in tscv.split(values):
        
        train_s = values.iloc[train]
        test_s = values.iloc[test]
        
        if scale:
            scaler = StandardScaler()
            scaler.fit(values.iloc[train])

            train_s = scaler.transform(values.iloc[train])
            test_s = scaler.transform(values.iloc[test])

        model.fit(train_s, y.iloc[train])
        pred = model.predict(test_s)
        pred = np.squeeze(np.asarray(pred))
        
        y_list.append(y.iloc[test])
        pred_list.append(pred)
    plt.plot(pred, label='predictions')
    plt.plot(y.iloc[test].to_numpy(), label='true')
    plt.xlabel('sample')
    plt.ylabel('value')
    plt.legend()
    plt.show()
    
    mape, wape, mse = metrics(y_list, pred_list, one_model=False)
    
    # There is some troubles with wape, sometimes wape equals to nan
    
    return  mape, np.nan_to_num(wape), mse


def timeseriesCVscore_catboost_grid(data, model, grid_params, scale=True):
    
    y_list = []
    pred_list = []
    
    data, y = preprocessing(data)
    

    values = data

    tscv = TimeSeriesSplit(n_splits=3, test_size=4) 

    for train, test in tscv.split(values):
        
        train_s = values.iloc[train]
        test_s = values.iloc[test]
        
        if scale:
            scaler = StandardScaler()
            scaler.fit(values.iloc[train])

            train_s = scaler.transform(values.iloc[train])
            test_s = scaler.transform(values.iloc[test])

        grid = CatBoostRegressor(logging_level='Silent', task_type="GPU",
                           devices='0:1',  allow_const_label=True)
        grid.grid_search(grid_params, train_s, y.iloc[train],  shuffle=False)
        pred = grid.predict(test_s)
        pred = np.squeeze(np.asarray(pred))
        
        y_list.append(y.iloc[test])
        pred_list.append(pred)
    
    
    mape, wape, mse = metrics(y_list, pred_list, one_model=False)
    
    # There is some troubles with wape, sometimes wape equals to nan
    
    return  mape, np.nan_to_num(wape), mse



    