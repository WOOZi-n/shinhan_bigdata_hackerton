# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:52:31 2022

@author: jmjwj
"""

#기본데이터

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
from sklearn.metrics import roc_curve, recall_score
from sklearn.ensemble import RandomForestClassifier


os.chdir("C:\\Users\\jmjwj\\Desktop\\shinhan")
os.getcwd()
rawdata = pd.read_csv('data_061.csv', encoding = 'euc_kr')


'''
분석대상에서 제외할 변수.
1. 활동성향과 크게 관련이 없는 변수
2. 너무나 일반적인 변수 ex) 식료품
3. 

분석대상변수
1. 이용자의 성향이 드러나는 변수
2. 이외 빈도수 상위 섹터
3. 이벤트의 발생을 암시하는 변수
4. 일정수준 이상의 금액이 의미를 주는 변수
'''

rawdata['travel'] = rawdata[['B1','B2','B3','B4','B5','B6','B7','B118']].sum(axis = 1)
rawdata['transport'] = rawdata[['B10','B11']].sum(axis = 1)
rawdata['shopping'] = rawdata[['B13','B14','B21','B31','B32', 'B72','B74','B75','B76','B77','B78']].sum(axis = 1)
rawdata['internet'] = rawdata[['B33','B34']].sum(axis = 1)
rawdata['eat'] = rawdata[['B35','B36','B37','B38','B39','B40','B41','B42']].sum(axis = 1)
rawdata['interior'] = rawdata[['B44','B46','B47','B48','B49','B52','B53']].sum(axis = 1)
rawdata['child'] = rawdata[['B68','B94','B155','B156','B88','B122','B123']].sum(axis = 1)
rawdata['leisure'] = rawdata[['B86','B87','B89','B90','B97','B98','B99','B100',
                              'B101','B102','B103','B104','B105']].sum(axis = 1)
rawdata['insurance'] = rawdata[['B106','B107']].sum(axis = 1)
rawdata['marry'] = rawdata[['B109','B110','B111','B115']].sum(axis =1 )
rawdata['funeral'] = rawdata[['B112','B117','B116']].sum(axis = 1)
rawdata['medical'] = rawdata[['B139','B140','B141','B142','B143','B144','B145','B146','B147','B148','B149']].sum(axis =1 )


sectorized = pd.concat([rawdata.iloc[:, :7], rawdata['travel'],
                        rawdata['transport'], rawdata['shopping'],
                        rawdata['internet'], rawdata['eat'], rawdata['interior'],
                        rawdata['child'], rawdata['leisure'], rawdata['insurance'],
                        rawdata['marry'], rawdata['funeral'], rawdata['medical'],
                        rawdata.iloc[:,18], rawdata.iloc[:,24], rawdata.iloc[:, 34],
                        rawdata.iloc[:,173:181]], axis = 1)



# X = sectorized.drop('P5', axis = 1)
# y = sectorized.P5

# 전처리
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y, random_state = 0)
personal_train, personal_test = X_train.iloc[:,:6], X_test.iloc[:,:6]
pay_train, pay_test = X_train.iloc[:,6:-6], X_test.iloc[:,6:-6]
# pattern_train, pattern_test = X_train.iloc[:,-6:], X_test.iloc[:,-6:]

# smote 이용 upsampling

#1. paydata

from imblearn.over_sampling import SMOTE

scaler = MinMaxScaler()
pay_train_scaled = scaler.fit_transform(pay_train)
pay_test_scaled = scaler.transform(pay_test)
sm = SMOTE(random_state = 0)
pay_train_resampled, y_train_resampled = sm.fit_resample(pay_train_scaled, y_train)

pay_train_resampled = pd.DataFrame(pay_train_resampled)
pay_train_resampled.columns = pay_train.columns
pay_test_scaled = pd.DataFrame(pay_test_scaled)
pay_test_scaled.columns = pay_train.columns


    # paydata 변수 변환
for X in [pay_train_resampled, pay_test_scaled]:        
    X['agri'] = X['B18'].where(X['B18'] == 0, 1)
    X['retail'] = X['B28'].where(X['B28'] == 0, 1)
    X['marry'] = X['marry'].where(X['marry'] < 100000, 1)
    X['marry'] = X['marry'].where(X['marry'] == 1, 0)
    X['funeral'] = X['funeral'].where(X['funeral'] < 500000, 1)
    X['funeral'] = X['funeral'].where(X['funeral'] == 1, 0)
    X.drop(['B18','B28'], axis = 1, inplace = True)
    

#로지스틱
pay_train_resampled.columns
log_reg = LogisticRegression()
log_reg = log_reg.fit(pay_train_resampled, y_train_resampled)
fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(pay_test_scaled)[:,1])
plt.plot(fpr, tpr)

pay_train_resampled = pay_train_resampled[['travel', 'shopping', 'leisure', 'agri', 'retail', 'marry', 'funeral', 'B167']]
pay_test_scaled= pay_test_scaled[['travel', 'shopping', 'leisure', 'agri', 'retail', 'marry', 'funeral', 'B167']]



# 변수 모두 포함시키는 게 낫다.

# log_reg2 = LogisticRegression()
# log_reg2 = log_reg2.fit(pay_train_resampled, y_train_resampled)
# fpr, tpr, thresholds = roc_curve(y_test, log_reg2.predict_proba(pay_test_scaled)[:,1])
# plt.plot(fpr, tpr)

# 랜덤포레스트

forest = RandomForestClassifier(max_depth = 1)
forest.fit(pay_train_resampled, y_train_resampled)
np.mean(y_test == forest.predict(pay_test_scaled))
recall_score(y_test, forest.predict(pay_test_scaled))
# 70퍼 수준에서 33퍼민감도



forest = RandomForestClassifier(max_depth = 1)
forest.fit(pay_train_resampled, y_train_resampled)
np.mean(y_test == forest.predict(pay_test_scaled))
recall_score(y_test, forest.predict(pay_test_scaled))
#57퍼 수준에서 55퍼 민감도

np.mean((log_reg.predict_proba(pay_test_scaled)[:,-1] > 0.5) == y_test)
recall_score((log_reg.predict_proba(pay_test_scaled)[:,-1] > 0.5) , y_test)
# 64수준서 12퍼 민감도

#랜덤포레스트가 더 낫다.




#2. personal_data
personal = pd.concat([personal_train, y_train], axis = 1)
personal_train_rate = personal.groupby(['P1','P2','P3','P4','P6']).P5.mean().reset_index()
personal_train_rate
personal_train_count =personal.groupby(['P1','P2','P3','P4','P6']).apply(lambda x : len(x)).reset_index()
personal_train_data = pd.concat([personal_train_rate, personal_train_count.iloc[:,-1]], axis = 1)

personal_train_data.columns = pd.Index(['P1', 'P2', 'P3', 'P4', 'P6', 'P5', 'count'], dtype='object')

group_rate = personal_train_data.groupby(['P3','P4','P6']).P5.transform('mean')

personal_train_data.P5.loc[personal_train_data['count'] < 20] = group_rate.loc[personal_train_data['count'] < 20]

# 데이터가 부족한 값은 p3p4p6평균치로 대체하였다
# 이 비율 값을 최종모델의 한 변수로 대입한다. 이로써 개인신상지수를 도출했다.
# 증권사 정보는 반영하지 않았다.(적합도수가 너무 낮아져서)



#3.pattern data 역시 upsampling 후에 모델 생성하겠다.

pattern_data = sectorized.iloc[:, -6:]
pattern_data = pd.concat([pattern_data, sectorized.P5], axis = 1)
recent_data = pattern_data.apply(lambda x : x % 10)

decay = 0.9
def moveaverage(x, decay):
    x = list(map(int, str(x)))
    sum = 0
    for i in range(len(x)):
        sum += decay**(len(x)-i-1)*x[i]
    return sum

def make_pattern_data(decay):
    pattern_ma = pattern_data.applymap(lambda x : moveaverage(x, decay)).drop('P5', axis = 1)
    pattern_train, pattern_test, y_train, y_test = train_test_split(pattern_ma, sectorized.P5, random_state = 0) 
    sm = SMOTE(random_state = 0)
    pattern_train_resampled, y_train_resampled = sm.fit_resample(pattern_train, y_train)
    return pattern_train_resampled, y_train_resampled

# 로지스틱
log_reg_pattern = LogisticRegression()
log_reg_pattern.fit(pattern_train_resampled, y_train_resampled)
recall_score(y_test, log_reg_pattern.predict(pattern_test))
log_reg_pattern.predict(pattern_test).sum()
y_test.sum()
np.mean(log_reg_pattern.predict(pattern_test) == y_test)

for decay in [0.8, 0.9, 1]:
    X_train, y_train = make_pattern_data(decay)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # fpr, tpr, threshold = roc_curve(y_test, model.predict_proba(pattern_test)[:,1])
    # plt.plot(fpr, tpr)
    print(model.predict_proba(pattern_test)[:,1])

#roc 곡선은 거의 차이가 없다. 최종모델 수준에서 다시 한번 생각해보자.

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

def preprocess_pay():
def preprocess_pattern():



def transform_final_data(rawdata):
    X = rawdata # P5 포함시킴 일단.
    y = rawdata.P5
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 0)
    personal_train, personal_test = X_train.iloc[:,:6], X_test.iloc[:,:6] # P7 뺀다 일단.
    pay_train, pay_test = X_train.iloc[:,7:-6], X_test.iloc[:,7:-6]
    pattern_train, pattern_test = X_train.iloc[:,-6:], X_test.iloc[:,-6:]
    data= pd.concat([personalpreprocess_personal(personal_train, personal_test, y),
                     preprocess_pay(personal_train, personal_test, y),
                     preprocess_pattern(personal_train, personal_test, y)], axis = 1)
    
    