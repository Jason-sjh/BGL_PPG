# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 
import os
import yaml 
import pickle
import argparse
from sklearn.preprocessing import StandardScaler
import scipy.signal as signal
from datetime import datetime
import json
import sys

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



from utils.feature_extraction import feature_extraction, apply_mav_filter_2d
from utils.plot_ceg import corr_plot, clarke_error_grid, sns_pair_plot

from module.PPG_model import PPG_model
from module.PPG_residual_model import PPG_Res_Model
from module.seq_script import train, validate, validate_result


def reverse_median(df):
    mid_point = np.median(df)
    df_r = 2* mid_point  - df 
    return df_r

def choice_ch(dict_ppg):
    print('-------- Choose Channel ')   
    dict_ppg_ch1 = {}
    
    for i, v in enumerate(dict_ppg):

        if dict_ppg[v].iloc[:,1].isnull().sum() == 0:
            
            if len(dict_ppg[v].iloc[:,0].value_counts()) > len(dict_ppg[v].iloc[:,1].value_counts()):
                dict_ppg_ch1[v] = dict_ppg[v].iloc[:,0]
            elif len(dict_ppg[v].iloc[:,0].value_counts()) < len(dict_ppg[v].iloc[:,1].value_counts()):
                dict_ppg_ch1[v] = dict_ppg[v].iloc[:,1]
            else:
                dict_ppg_ch1[v] = dict_ppg[v].iloc[:,0]
        else:
            dict_ppg_ch1[v] = dict_ppg[v].iloc[:,0]

    ppg_np_data = np.empty((1,1024))

    for i,v in enumerate(dict_ppg_ch1):

        ch1_data = dict_ppg_ch1[v].values.reshape(-1,1024)
        ppg_np_data = np.append(ppg_np_data, ch1_data, axis = 0)

    ppg_np_data = ppg_np_data[1:,:]

    # ( sample_len , 1024 )
        
    return ppg_np_data

def read_ppg_reverse(data_path = 'data'):
    print('-------- Data read CSV and reverse data')
    subject_lis = os.listdir(data_path)
    dict_a = {}
    dict_a_new = {}
    label_data = []
    for i, n in enumerate(subject_lis):
        folder_path = data_path + '/' + n
        xlsx_file_names = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        per_bgl = pd.read_excel(folder_path + '/'+ xlsx_file_names[0], skiprows = 2, engine='openpyxl')
        per_bgl.drop(per_bgl.columns[0], axis=1, inplace=True)

        label_data = label_data + list(per_bgl.iloc[:,3].values)
        csv_lis = os.listdir(folder_path + '/Data')
        for i in range(len(csv_lis)):
            data_bgl = pd.read_csv(folder_path + '/Data/' + csv_lis[i], encoding='cp949', skiprows = 6, names = ['ch1','ch2' ] , dtype={'ch1': float, 'ch2':float})
            data_bgl = data_bgl[:1024]
            dict_a[csv_lis[i]] = data_bgl        
        for i in range(len(per_bgl)):
            if (per_bgl.iloc[i, 4] == '반전') | (per_bgl.iloc[i, 6] == '반전'):
                file_name = per_bgl.iloc[i, 2]
                df = dict_a[file_name]
                if (per_bgl.iloc[i, 4] == '반전'):
                    # print(df.iloc[:,0])
                    df_new = reverse_median(df.iloc[:,0])
                    df.iloc[:,0] = df_new
                if (per_bgl.iloc[i, 6] == '반전'):

                    df_new = reverse_median(df.iloc[:,1])
                    df.iloc[:,1] = df_new
                dict_a_new[file_name] = df

            else:
                file_name = per_bgl.iloc[i, 2]
                dict_a_new[file_name] = dict_a[file_name]    


    return dict_a_new ,label_data


def tuning_script(X_train, X_test,y_train, y_test):

    print('------------------------ Tuning ML model with Gridsearch ') 
    
    svm_param_grid = {'C': [0.1, 1, 10, 100, 1000], 'epsilon': [0.01, 0.1, 0.5, 1], 'kernel': ['poly', 'rbf', 'sigmoid']}
    gbm_param_grid = {'n_estimators': [50, 100, 200, 300, 400],
                 "learning_rate": [0.001 ,0.01, 0.025, 0.05, 0.1],
                  'max_depth': [3, 5, 7, 9],
                 "subsample":[0.5, 0.8, 0.9, 1.0]}

    rf_param_grid = {'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 10, 20, 30],
                 'min_samples_split': [2, 5, 10, 20],'min_samples_leaf': [1, 2, 4, 8]}

    lr_param_grid = {'fit_intercept': [True, False],
                 'normalize': [True, False],
                 'copy_X': [True, False],
                 'n_jobs': [None, 1, 2] }

    xgb_param_grid = {
                    'learning_rate': [0.001, 0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 300, 500],
                     'max_depth': [5, 10, 20, 25],
                     'subsample': [0.7, 0.9, 1.0],
                     'colsample_bytree': [0.6, 0.8, 1.0],
                     'reg_alpha': [0, 0.001, 0.01, 0.1],
                     'reg_lambda': [ 0.001, 0.01, 0.1] }

    cv_num = 3
    svm_grid_search = GridSearchCV(SVR(), svm_param_grid, cv=cv_num)
    gbm_grid_search = GridSearchCV(GradientBoostingRegressor(), gbm_param_grid, cv=cv_num)
    rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=cv_num)
    lr_grid_search = GridSearchCV(LinearRegression(), lr_param_grid, cv=cv_num)
    xgb_grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), xgb_param_grid, cv=cv_num)

    print('------------------------ Tuning SVM ')        
    svm_grid_search.fit(X_train, y_train)
    print('------------------------ Tuning GBM ')    
    gbm_grid_search.fit(X_train, y_train)
    print('------------------------ Tuning RF ')    
    rf_grid_search.fit(X_train, y_train)
    print('------------------------ Tuning LR ')    
    lr_grid_search.fit(X_train, y_train)
    print('---------------- Tuning XGB ')    
    xgb_grid_search.fit(X_train, y_train)    
    
    print("Best Parameters for Linear Regression:", lr_grid_search.best_params_)
    print("Best Parameters for SVR:", svm_grid_search.best_params_)
    print("Best Parameters for Gradient Boosting:", gbm_grid_search.best_params_)
    print("Best Parameters for Random Forest:", rf_grid_search.best_params_)
    print("Best Parameters for XGB :", xgb_grid_search.best_params_)
    	
    best_parameter_dict = {'SVM' : 	svm_grid_search.best_params_ , 'LR': lr_grid_search.best_params_, 'GBM': gbm_grid_search.best_params_, 'RF': rf_grid_search.best_params_, 'XGB': xgb_grid_search.best_params_  }
    	
    with open('./config/ml_best_parameter.yaml', 'w') as file:   
        yaml.dump(best_parameter_dict, file)
    
    return best_parameter_dict   


def fit_pred(model, X_train, X_test,y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f' {model.__class__.__name__} mARD : {mARD(y_test, pred)}')
    print(f' {model.__class__.__name__} MAE : {mean_absolute_error(y_test, pred)}')   
    print(f' {model.__class__.__name__} RMSE : {np.sqrt(mean_squared_error(y_test, pred))}')   
    
    return pred


def train_ml(X_train, X_test,y_train, y_test, args):
    print('------------------------ Train ML model ')           
    parameter_path = './config/ml_best_parameter.yaml'
    hyper_dict={}
    if args.best_parameter:
        if not os.path.exists(parameter_path):
            print(' f{parameter_path} :: Best Parameter is not exist ! ')
            sys.exit()
        else :
            with open('./config/ml_best_parameter.yaml', 'r') as file:
                loaded_dict = yaml.safe_load(file)
            hyper_dict = loaded_dict   
            
    else : 
        if args.tuning:
            hyper_dict = tuning_script(X_train, X_test,y_train, y_test)            
        elif not args.tuning:   
            hyper_dict = {'SVM':{} ,'GBM':{}, 'RF':{}, 'LR':{}, 'XGB':{}}
            

    svm_model = SVR(**hyper_dict['SVM'])
    gbm_model = GradientBoostingRegressor(**hyper_dict['GBM'])
    rf_model = RandomForestRegressor(**hyper_dict['RF'])
    lr_model = LinearRegression(**hyper_dict['LR'])
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **hyper_dict['XGB'])
    
    model_list = [svm_model, gbm_model, rf_model, lr_model, xgb_model]
    
    train_result =[]
    for i in range(len(model_list)):
         train_result.append(fit_pred(model_list[i], X_train, X_test,y_train, y_test))
         print('\n')
         

    save_result(y_test, model_list ,train_result)
    
    for i in range(len(model_list)):
        clarke_error_grid(y_test, train_result[i], model_list[i].__class__.__name__ )
    

def save_result(y_test, model_list ,train_result):
    mARD_lis =[]
    mae_lis =[]
    rmse_lis=[]
    model_name = []    
    if not isinstance(model_list, list):
        model_list = [model_list]
    if not isinstance(train_result, list):
        train_result = [train_result]
    
    for i in range(len(model_list)):
        mARD_lis.append(  round(float(mARD(y_test, train_result[i])) , 4)  )
        mae_lis.append(  round(float(mean_absolute_error(y_test, train_result[i])) , 4) )
        rmse_lis.append( round(float(np.sqrt(mean_squared_error(y_test, train_result[i]))) , 4) )
        model_name.append(  model_list[i].__class__.__name__)


    all_data = {
       "Model":model_name,
       "mARD": mARD_lis,
       "MAE": mae_lis,
       "RMSE": rmse_lis
    }

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")       
    with open(f"./result/result_{current_time}.json", 'w') as json_file:
        json.dump(all_data, json_file)



def mARD(ref,pred):
    mARD_val = (np.sum(np.abs(pred - ref ) / ref) *100 )/len(pred)    
    return mARD_val


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default = './data')
    parser.add_argument("--config", default = './config/baseline.yaml', action='store')
    parser.add_argument("--best_parameter", default = False, action='store_true')
    parser.add_argument("--eda", default = False, action='store_true')
    parser.add_argument("--tuning", default = False, action='store_true')
    parser.add_argument("--deep_learning", default = False, action='store_true')

    args = parser.parse_args()
    
    with open(f'{args.config}', 'r') as file:
        deep_config = yaml.load(file, Loader=yaml.FullLoader)

    
    dict_ppg, label_data = read_ppg_reverse(data_path = args.data_path)
    ppg_data = choice_ch(dict_ppg)
    label_data  = np.array(label_data)

    #### Machine Learning 
    if not args.deep_learning:
        result_feature = feature_extraction(ppg_data)  
        if args.eda:
            corr_plot(result_feature, label_data, folder_path = './result/EDA')
            sns_pair_plot(result_feature, label_data,folder_path = './result/EDA')
        
        ## scaler 
        scaler = StandardScaler()
        scaled_result_data = scaler.fit_transform(result_feature)

        ## Split train test   
        bins = np.linspace(min(label_data), max(label_data), num=5)
        bin_labels = np.digitize(label_data, bins)

        stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in stratified_splitter.split(scaled_result_data, bin_labels):
            X_train, X_test = scaled_result_data[train_index], scaled_result_data[test_index]
            y_train, y_test = label_data[train_index], label_data[test_index]

        ## just split    
        # X_train, X_test, y_train, y_test = train_test_split(scaled_result_data, label_data, test_size=0.2, random_state=42)

        train_ml(X_train, X_test,y_train, y_test, args)
     
       
    #### Deep learning 
    elif args.deep_learning:
        ppg_data = apply_mav_filter_2d(ppg_data , 10)
        scaler = StandardScaler()
        ppg_data = scaler.fit_transform(ppg_data)
        X = torch.from_numpy(ppg_data)
        y = torch.from_numpy(label_data)
        if y.ndim > 1:
            y = y.view(-1)
    
        bins = np.linspace(min(y), max(y), num=5)
        bin_labels = np.digitize(y, bins)

        stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in stratified_splitter.split(X, bin_labels):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_test, y_test)


        train_loader = DataLoader(train_dataset, batch_size = deep_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size =deep_config['batch_size'])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        if deep_config['model_type'] == 1:
            model = PPG_model().to(device)
        elif deep_config['model_type'] == 2:
            model = PPG_Res_Model().to(device)

        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=deep_config['learning_rate'])
        
        num_epochs = deep_config['num_epoch']  
        
        for epoch in range(num_epochs):
            avg_train_loss = train(model, device, train_loader, optimizer, criterion)
            avg_valid_loss = validate(model, device, val_loader, criterion)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

       
        ref, pred = validate_result(model, device, val_loader, criterion)
        save_result(ref, model, pred )
        clarke_error_grid(ref, pred, model.__class__.__name__ )        
        
        


