# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 18:09:23 2022

@author: jmjwj
"""
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import statsmodels.api as stm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


os.chdir("C:\\Users\\jmjwj\\Desktop\\shinhan")
os.getcwd()
rawdata = pd.read_csv('data_061.csv', encoding = 'euc_kr')


## 최종모델: 
    # 변수 : 신상데이터 확률값, paydata 로지스틱 확률 값, 패턴데이터 확률 값, A 증권사 더미변수

# train 데이터로만 학습해야하는 것 명심.
def preprocess_personal(personal_train, personal_test):
    personal_train_rate = personal_train.groupby(['P1','P2','P3','P4','P6']).P5.mean().reset_index()
    personal_train_count =personal_train.groupby(['P1','P2','P3','P4','P6']).apply(lambda x : len(x)).reset_index()
    personal_train_data = pd.concat([personal_train_rate, personal_train_count.iloc[:,-1]], axis = 1)
    personal_train_data.columns = pd.Index(['P1', 'P2', 'P3', 'P4', 'P6', 'P5', 'count'], dtype='object')
    group_rate = personal_train_data.groupby(['P3','P4','P6']).P5.transform('mean')
    personal_train_data.P5.loc[personal_train_data['count'] < 20] = group_rate.loc[personal_train_data['count'] < 20]
    personal_train_processed = pd.merge(personal_train.reset_index(), personal_train_data, on = ['P1','P2','P3','P4','P6'])
    personal_test_processed = pd.merge(personal_test.reset_index(), personal_train_data, on = ['P1','P2','P3','P4','P6'])
    return personal_train_processed[['index','P5_y']], personal_test_processed[['index','P5_y']]    

def sectorized_pay(X): 
        X['travel'] = X[['B1','B2','B3','B4','B5','B6','B7','B118']].sum(axis = 1)
        X['transport'] = X[['B10','B11']].sum(axis = 1)
        X['shopping'] = X[['B13','B14','B21','B31','B32', 'B72','B74','B75','B76','B77','B78']].sum(axis = 1)
        X['internet'] = X[['B33','B34']].sum(axis = 1)
        X['eat'] = X[['B35','B36','B37','B38','B39','B40','B41','B42']].sum(axis = 1)
        X['interior'] = X[['B44','B46','B47','B48','B49','B52','B53']].sum(axis = 1)
        X['child'] = X[['B68','B94','B155','B156','B88','B122','B123']].sum(axis = 1)
        X['leisure'] = X[['B86','B87','B89','B90','B97','B98','B99','B100',
                                      'B101','B102','B103','B104','B105']].sum(axis = 1)
        X['insurance'] = X[['B106','B107']].sum(axis = 1)
        X['marry'] = X[['B109','B110','B111','B115']].sum(axis =1 )
        X['funeral'] = X[['B112','B117','B116']].sum(axis = 1)
        X['medical'] = X[['B139','B140','B141','B142','B143','B144','B145','B146','B147','B148','B149']].sum(axis =1 )
        X_new = X[['travel','transport', 'shopping', 'internet', 'eat', 'interior', 'child', 'leisure',
        'insurance', 'marry', 'funeral', 'medical', 'B18', 'B28','B167']]
        X_new['agri'] = X_new['B18'].where(X_new['B18'] == 0, 1)
        X_new['retail'] = X_new['B28'].where(X_new['B28'] == 0, 1)
        X_new['marry'] = X_new['marry'].where(X_new['marry'] < 100000, 1)
        X_new['marry'] = X_new['marry'].where(X_new['marry'] == 1, 0)
        X_new['funeral'] = X_new['funeral'].where(X_new['funeral'] < 500000, 1)
        X_new['funeral'] = X_new['funeral'].where(X_new['funeral'] == 1, 0)
        X_new.drop(['B18','B28'], axis = 1, inplace = True)
        return X_new
     
    
def preprocess_pay(pay_train, pay_test, y_train):
    #우선 upsampling
    train_set, test_set = sectorized_pay(pay_train), sectorized_pay(pay_test)
    sm = SMOTE(random_state = 0)
    X_train_resampled, y_train_resampled = sm.fit_resample(train_set, y_train)
    log_reg = LogisticRegression(max_iter = 10000000)
    log_reg.fit(X_train_resampled, y_train_resampled)
    train_pred = pd.DataFrame(log_reg.predict_proba(train_set)[:,1]) 
    test_pred = pd.DataFrame(log_reg.predict_proba(test_set)[:,1])
    train_pred.index=  train_set.index
    test_pred.index= test_set.index
    return train_pred, test_pred

def moveaverage(x, decay):
    x = list(map(int, str(x)))
    sum = 0
    for i in range(len(x)):
        sum += decay**(len(x)-i-1)*x[i]
    return sum

# def make_pattern_data(data, decay):
#     pattern_ma = data.applymap(lambda x : moveaverage(x, decay))
#     pattern_train, pattern_test, y_train, y_test = train_test_split(pattern_ma, sectorized.P5, random_state = 0) 
#     sm = SMOTE(random_state = 0)
#     pattern_train_resampled, y_train_resampled = sm.fit_resample(pattern_train, y_train)
#     return pattern_train_resampled, y_train_resampled

def preprocess_pattern(pattern_train, pattern_test, y_train, decay):
    X_train = pattern_train.applymap(lambda x : moveaverage(x, decay))
    X_test = pattern_test.applymap(lambda x : moveaverage(x, decay))
    sm = SMOTE(random_state = 0)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
    log_reg = LogisticRegression()
    log_reg.fit(X_train_resampled, y_train_resampled)
    train_pred = pd.DataFrame(log_reg.predict_proba(X_train)[:,1])
    test_pred = pd.DataFrame(log_reg.predict_proba(X_test)[:,1])
    train_pred.index =  X_train.index
    test_pred.index = X_test.index
    return train_pred, test_pred

def transform_final_data(rawdata):
    x = rawdata # P5 포함시킴 일단.
    y = rawdata.P5
    X_train, X_test, y_train, y_test = train_test_split(x,y, random_state= 0)
    personal_train, personal_test = X_train.iloc[:,:6], X_test.iloc[:,:6] # P7 뺀다 일단.
    pay_train, pay_test = X_train.iloc[:,7:-6], X_test.iloc[:,7:-6]
    pattern_train, pattern_test = X_train.iloc[:,-6:], X_test.iloc[:,-6:]
    personal_train_processed, personal_test_processed = preprocess_personal(personal_train, personal_test)
    pay_train_processed, pay_test_processed = preprocess_pay(pay_train, pay_test, y_train)
    pattern_train_processed, pattern_test_processed = preprocess_pattern(pattern_train, pattern_test, y_train, decay = 0.9)
    final_train = pd.merge(pd.merge(personal_train_processed, pay_train_processed.reset_index(), on = 'index'), 
                           pattern_train_processed.reset_index(), on = 'index')
    final_test = pd.merge(pd.merge(personal_test_processed, pay_test_processed.reset_index(), on = 'index'), 
                           pattern_test_processed.reset_index(), on = 'index')

    final_train = pd.merge(final_train, pd.get_dummies(X_train.P7).reset_index(), on = 'index')
    final_test = pd.merge(final_test, pd.get_dummies(X_test.P7).reset_index(), on = 'index')
    
    
    return final_train.drop('index', axis = 1), final_test.drop('index', axis = 1), y_train, y_test
                            
final_train, final_test, y_train, y_test = transform_final_data(rawdata)

#최종모델
sm = SMOTE(random_state = 0)
final_train_resampled, y_train_resampled = sm.fit_resample(final_train, y_train)

#로지스틱
final_model = stm.GLM(y_train_resampled, final_train_resampled, family = stm.families.Binomial())
final_model = final_model.fit()
final_model.summary()
fpr, tpr, threshold = roc_curve(y_test, final_model.predict(final_test))
plt.plot(fpr, tpr)
np.mean(y_test == final_model.predict(final_test))
recall_score(y_test, final_model.predict(final_test))
precision_score(y_test, final_model.predict(final_test))

# 성능 좆구리네 _ 그냥 의미없는 모델이 되어버림...





