import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm

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

# 1. 구매 금액 중심 탐색
# 정규화필요
X = sectorized.iloc[:,7:-7]
y = sectorized.P5

X_new = np.array(X)
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)

dat = []
for k in range(3,25):
    model = KMeans(n_clusters = k)
    model.fit(X_new)
    dat.append(model.inertia_)

pd.DataFrame(dat).plot()
elbow = 18

kmeans = KMeans(n_clusters = 18, random_state = 0)
kmeans.fit(X_new)
X_cluster = pd.Series(kmeans.predict(X_new))

#클러스터별 비율 
sectorized.groupby(X_cluster).P5.mean().sort_values()
# 5-3-(8-2-6-12-16-1-17-4-10-0)-(15-7-11-13)-(9-14)

kmeans.cluster_centers_[[8,14,11,6, 10]]
kmeans.cluster_centers_[[15,7,11,13]]
## 극단적 클러스터

# 5클러스터 : 도매상
# 3클러스터 : 농업종사자
# 9클러스터 : 장례
# 14클러스터 : 인터넷 이용자

## 상위 클러스터

#15:레저에 많은비용
#7:결혼
#11:여행
#13:쇼핑



# 5클러스터
sectorized.loc[sectorized.B18 > 0].P5.mean()
sectorized.loc[sectorized.B18 > 0].B18.describe()
B18_group = pd.qcut(sectorized.loc[sectorized.B18 > 0].B18, 5)
sectorized.loc[sectorized.B18 > 0].groupby(B18_group).P5.mean().plot()
plt.xticks(fontsize = 10, rotation = 45)
plt.title('P5 ~ B18')
sectorized.loc[sectorized.B18 > 0].groupby(B18_group).P5.count()
# 도매점에서 구매금액이 있는 사람들은 낮고, 그 수가 일정 수치(대략 10만원)를 넘어갈 때 큰폭으로 비율이 감소.

# 3클러스터 
sectorized.loc[sectorized.B28 > 0].P5.mean()
sectorized.loc[sectorized.B28 > 0].B28.describe()
B28_group = pd.qcut(sectorized.loc[sectorized.B28 > 0].B28, 5)
sectorized.loc[sectorized.B28 > 0].groupby(B28_group).P5.mean()
sectorized.loc[sectorized.B28 > 0].groupby(B28_group).P5.count()
# 농민들은 증권이용확률이 낮다.

# 9클러스터 
sectorized.loc[sectorized.funeral > 0].P5.mean()
sectorized.loc[sectorized.funeral > 0].funeral.describe()
bins = [100000*i for i in range(10)]
fun_group = pd.cut(sectorized.loc[sectorized.funeral> 0].funeral, bins = 10)
sectorized.loc[sectorized.funeral > 0].groupby(fun_group).P5.mean().plot()
sectorized.loc[sectorized.funeral > 0].groupby(fun_group).P5.count()

# 장례 주최자와 참석자의 구분?
# 장례 소비가 50만원을 넘어가면서 유의미한 차이가 있음.

dat = []
for i in range(30):
    dat.append(sectorized.loc[sectorized.funeral > i*100000].P5.mean())    
pd.DataFrame(dat).plot()


# 14클러스터 
sectorized.loc[sectorized.internet > 0].P5.mean()
sectorized.loc[sectorized.internet> 0].B28.describe()
internet_group = pd.cut(sectorized.loc[sectorized.internet> 0].internet, 20)
sectorized.loc[sectorized.internet> 0].groupby(internet_group).P5.mean().plot()
sectorized.loc[sectorized.internet> 0].groupby(internet_group).P5.count()
# 인터넷 구매내역과 증권사 이용여부는 양의 상관관계가 있는 것으로 보임


#15:레저에 많은비용
sectorized.loc[sectorized.leisure > 0].P5.mean()
sectorized.loc[sectorized.leisure> 0].leisure.describe()
leisure_group = pd.qcut(sectorized.loc[sectorized.leisure> 0].leisure, 50)
sectorized.loc[sectorized.leisure> 0].groupby(leisure_group).P5.mean().plot()
sectorized.loc[sectorized.leisure> 0].groupby(leisure_group).P5.mean()
sectorized.loc[sectorized.leisure> 0].groupby(leisure_group).P5.count()
# 레저도 대략 30만원을 넘어가면서 상관관계가 있음


#7:결혼
sectorized.loc[sectorized.marry > 0].P5.mean()
sectorized.loc[sectorized.marry> 0].marry.describe()
marry_group = pd.cut(sectorized.loc[sectorized.marry> 0].marry, 20)
sectorized.loc[sectorized.marry> 0].groupby(marry_group).P5.mean().plot()
sectorized.loc[sectorized.marry> 0].groupby(marry_group).P5.mean()
sectorized.loc[sectorized.marry> 0].groupby(marry_group).P5.count()
##10만원 이상 시 선형관계 

#11:여행
sectorized.loc[sectorized.travel > 0].P5.mean()
sectorized.loc[sectorized.travel> 0].travel.describe()
travel_group = pd.qcut(sectorized.loc[sectorized.travel> 100000].travel, 20)
sectorized.loc[sectorized.travel> 0].groupby(travel_group).P5.mean().plot()
sectorized.loc[sectorized.travel> 0].groupby(travel_group).P5.mean()
sectorized.loc[sectorized.travel> 0].groupby(travel_group).P5.count()
#여행도 레저와 비슷한 양상

#13:쇼핑
sectorized.loc[sectorized.shopping > 0].P5.mean()
sectorized.loc[sectorized.shopping> 0].shopping.describe()
shopping_group = pd.qcut(sectorized.loc[sectorized.shopping> 0].shopping, 70)
sectorized.loc[sectorized.shopping> 0].groupby(shopping_group).P5.mean().plot()
sectorized.loc[sectorized.shopping> 0].groupby(shopping_group).P5.mean()
sectorized.loc[sectorized.shopping> 0].groupby(shopping_group).P5.count()
# 쇼핑그룹은 전체적으로 균일하게 확률이 높음.





#elbow = 18
kmean_rate = KMeans(n_clusters = 18, random_state= 0)
kmean_rate.fit(X_rate_notnull)
X_rate_cluster = pd.Series(kmean_rate.predict(X_rate_notnull))
sectorized[X_rate.notnull().all(axis = 1)].groupby(X_rate_cluster).P5.mean()
sectorized[X_rate.notnull().all(axis = 1)].groupby(X_rate_cluster).apply(lambda x: len(x))

###3. 차원축소
# from sklearn.decomposition import PCA
# pca = PCA(2)
# pca.explained_variance_ratio_
# X2D = pca.fit_transform(X_rate_notnull)
# plt.scatter(X2D[:,0], X2D[:,1], c = sectorized[X_rate.notnull().all(axis = 1)].P5, cmap =  'jet', alpha = 0.1)
# plt.xlim(0,2)
# plt.ylim(0,2)

###특징추출후, 로지스틱 회귀 돌리기
# 일반 로지스틱은 거의 의미가 없네
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,  y_test = train_test_split(X_new, y, random_state = 0)

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
log_reg.predict(X_test)
log_reg.score(X_test, y_test)
pd.crosstab(log_reg.predict(X_test), y_test)
np.mean(y_test == 0)

#단순 랜덤포레스트도 마찬가지
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 500,max_depth = 2, n_jobs = -1, random_state= 0)
forest.fit(X_train, y_train)
pd.crosstab(forest.predict(X_test), y_test)

# 특정 특징관해서 모델 생성해야 함.


## 총 지출과 쇼핑, 레저, 여행, 인터넷 섹터와의 상관관계 파악하자.

sectorized.loc[sectorized.B167 > 0].P5.mean()
sectorized.loc[sectorized.B167> 0].B167.describe()
group = pd.qcut(sectorized.loc[sectorized.B167> 100000].B167, 50)
sectorized.loc[sectorized.B167> 0].groupby(group).P5.mean().plot()
sectorized.loc[sectorized.B167> 0].groupby(group).P5.mean()
sectorized.loc[sectorized.B167> 0].groupby(group).P5.count()

# 총 지출과 주식 여부는 상관관계가 있다.
# 그렇다면, 쇼핑, 레저, 여행, 인터넷 섹터의 결과는 총 지출이 높은데에 의한 효과일 수도 있다.
# 일단 상관성을 파악해보자.

plt.scatter(sectorized.B167, sectorized.shopping, c = sectorized.P5, cmap = 'jet', alpha = 0.2, s= 5)
plt.xlim(0,100000000)
#쇼핑그룹은 해석이 애매하네

plt.scatter(sectorized.B167, sectorized.leisure, c = sectorized.P5, cmap = 'jet', alpha = 0.2, s= 5)
plt.xlim(0,100000000)


plt.scatter(sectorized.B167, sectorized.travel, c = sectorized.P5, cmap = 'jet', alpha = 0.2, s= 5)
plt.xlim(0,100000000)

plt.scatter(sectorized.B167, sectorized.internet, c = sectorized.P5, cmap = 'jet', alpha = 0.2, s= 5)
plt.xlim(0,100000000)




# 그럼, 다시 비율차원 분석으로 들어가보자.

## 2 : 구매 비율 중심 탐색. B167이 0이면 제외해야 함
X_rate = sectorized.iloc[:, 7:-6].loc[sectorized.B167 > 0]
X_rate = X_rate.apply(lambda x : x / X_rate.B167)
X_rate1 = pd.concat([X_rate, sectorized.loc[sectorized.B167 > 0].P5], axis = 1)

shopper = X_rate1.loc[X_rate1.shopping > 0]

shopper_group = pd.cut(shopper.shopping, 10)
shopper.groupby(shopper_group).P5.mean().plot()
shopper.groupby(leisure_group).P5.count()

### 쇼핑금액 비율이 높은 사람들은 주식참여와 음의관계
leisurer = X_rate1.loc[X_rate1.leisure > 0]
leisure_group = pd.cut(leisurer.leisure, 10)
shopper.groupby(leisure_group).P5.mean().plot()
shopper.groupby(leisure_group).P5.count()

#레저비율은 주식참여와 양의 관계
leisurer = X_rate1.loc[X_rate1.leisure > 0]
leisure_group = pd.qcut(leisurer.leisure, 10)
shopper.groupby(leisure_group).P5.mean().plot()
shopper.groupby(leisure_group).P5.count()

#여행 주식참여와 양의 관계
traveler = X_rate1.loc[X_rate1.travel > 0]
traveler_group = pd.cut(traveler.travel[traveler.travel < 1], 10)
traveler.groupby(traveler_group).P5.mean().plot()
traveler.groupby(traveler_group).P5.count()

internetuser = X_rate1.loc[X_rate1.internet > 0]
internet_group = pd.cut(internetuser.internet[internetuser.internet< 1], 30)
internetuser.groupby(internet_group).P5.mean().plot()
internetuser.groupby(internet_group).P5.count()

## 인터넷 비율과 주식은 큰 연관성이 없다.


## 결론 ; 쇼핑은 주식과 음의 관계, 레저, 여행은 양의 관계, 인터넷비율은 큰 관련 없음. 즉 인터넷은 총 지출에 대한 부수효과일 가능성

### 요약 : 
    # 결제정보에서 두 가지 종류의 지수를 도출한다.
    #1. 이벤트 : 결혼, 장례  : 특정 금액 초과할 시 변수로 쓰고싶은데.. 정 안되면 더미변수
    #2. 소비패턴 : 레저, 여행, 쇼핑 : 이건 나름의 선형성이 있어보이니 로지스틱이 적절할듯.
    #3. 직업지 : 농민, 도매상 : 더미변수.
    #4. 총 결제 금액
    # 신상이나 패턴은 랜덤포레스트로 분석
    
# 일반 로지스틱 모델    

from sklearn.metrics import recall_score, confusion_matrix, roc_curve

X = sectorized.iloc[:, 7:-6]
log_reg = sm.Logit(y,X)
log_reg = log_reg.fit()

recall_score(y, log_reg.predict(X) > 0.1)
np.mean(y == (log_reg.predict(X) > 0.1))

log_reg.summary()

X['agri'] = X['B18'].where(X['B18'] == 0, 1)
X['retail'] = X['B28'].where(X['B28'] == 0, 1)
X['marry'] = X['marry'].where(X['marry'] < 100000, 1)
X['marry'] = X['marry'].where(X['marry'] == 1, 0)
X['funeral'] = X['funeral'].where(X['funeral'] < 500000, 1)
X['funeral'] = X['funeral'].where(X['funeral'] == 1, 0)



pay_data = X[['travel', 'shopping', 'leisure', 'agri', 'retail', 'marry', 'funeral', 'B167']]
pay_data = sm.add_constant(pay_data)
log_reg2 = sm.Logit(y,pay_data)
log_reg2 = log_reg2.fit()
log_reg2.summary()
recall_score(y, log_reg2.predict(pay_data) > 0.1)
np.mean(y == (log_reg2.predict(pay_data) > 0.1))
confusion_matrix(y, (log_reg2.predict(pay_data)> 0.1))

fpr1, tpr1, thresholds = roc_curve(y, log_reg.predict(X))
fpr2, tpr2, thresholds2 = roc_curve(y, log_reg2.predict(pay_data))

roc = [(fpr1, tpr1), (fpr2, tpr2)]
plt.plot(fpr1, tpr1, label = 'normal')
plt.plot(fpr2, tpr2, label = 'pay_data')
plt.legend()

#그래도 일반 로지스틱보단 내가 만든게 조금 더 낫다.

# ㅇㅋ. 이제 랜덤포레스트 이용한 기본정보 분석

personal_data = sectorized[['P1', 'P2', 'P3', 'P4','P5', 'P6','P7']]
personal_data.groupby(personal_data.P1).P5.mean() # 남녀비율 비슷
personal_data.groupby(personal_data.P2).P5.mean() # 30대에서 높은확률
personal_data.groupby(personal_data.P3).P5.mean() # 은행활동고객 확률 높다
personal_data.groupby(personal_data.P4).P5.mean() # 카드우수고객 높다
personal_data.groupby(personal_data.P6).P5.mean() # 라이프 활동고객 높다
personal_data.groupby(personal_data.P7).P5.mean() # 증권사 활동고객은 압도적.



personal_group = personal_data.groupby([personal_data.P1,personal_data.P2,
                       personal_data.P3,personal_data.P4,
                       personal_data.P6, personal_data.P7]).P5.apply(lambda x: [x.mean(), x.count()]).reset_index()

mean = []
for i in range(len(personal_group.P5)):
    mean.append(personal_group.P5[i][0])
personal_group['mean'] = mean

count = []
for i in range(len(personal_group.P5)):
    count.append(personal_group.P5[i][1])
personal_group['count'] = count

personal_group.drop('P5', axis = 1, inplace= True)
personal_group_over100 = personal_group.loc[personal_group['count'] > 100]
personal_group_over100.sort_values(by = 'mean', ascending = False)

personal_group_without_P7 = personal_data.groupby([personal_data.P1,personal_data.P2,
                       personal_data.P3,personal_data.P4,
                       personal_data.P6]).P5.apply(lambda x: [x.mean(), x.count()]).reset_index()

mean = []
for i in range(len(personal_group_without_P7.P5)):
    mean.append(personal_group_without_P7.P5[i][0])
personal_group_without_P7['mean'] = mean

count = []
for i in range(len(personal_group_without_P7.P5)):
    count.append(personal_group_without_P7.P5[i][1])
personal_group_without_P7['count'] = count

personal_group_without_P7.drop('P5', axis = 1, inplace= True)
personal_group_without_P7_over100 = personal_group_without_P7.loc[personal_group_without_P7['count'] > 100]
personal_group_without_P7_over100.sort_values(by = 'mean', ascending = False)


forest = RandomForestClassifier(n_estimators = 500, max_depth = 1)

forest.fit(personal_data.drop('P5', axis = 1), personal_data.P5)
# 증권사 파워가 너무 셈. : 두 가지 모델로 비교하자. 1. 증권사는 무조건 1 / 2. 전체 변수 다 넣은 로지스틱 모델. 3. 전체 다넣은 랜덤포레스트 모델

##증권사 이용객중 주식투자 안하는 사람들?

personal_data.loc[personal_data.P7 == 'A증권사'].loc[personal_data.P5 == 0].groupby([personal_data.P1,personal_data.P2,
                                                                                  personal_data.P3,personal_data.P4,
                                                                                  personal_data.P6]).P5.count().reset_index().sort_values(by = 'P5', ascending = False)

from sklearn.manifold import TSNE
tsne = TSNE()
person_tsne = tsne.fit_transform(personal_group_without_P7[['P3','P4','P6']])
sns.scatterplot(person_tsne[:, 0], person_tsne[:,1], c = personal_group.P5)
# 좀 시각화를 해야하는데

#이건 이산자료 분석으로 가야할듯. 일단 A증권사 이용자는 무조건 1로 예측하거나/차후 로지스틱 모델에 독립적으로 넣기.
# P1-P4, P6 과 P5의 관계는 이산자료 분석 책에서 아이디어를 가져오자.


# 다음은 패턴정보 분석. 최근 한달 숫자와 전체 1 합을 생각하자.
pattern_data = sectorized.iloc[:, -6:]

pattern_data = pd.concat([pattern_data, sectorized.P5], axis = 1)
recent_data = pattern_data.apply(lambda x : x % 10)

recent_data.groupby('E1').P5.mean() # E1이 1이면 비율이 낮다
recent_data.groupby('E1').P5.count() 
recent_data.groupby('E2').P5.mean() # E2이 1이면 비율이 낮다
recent_data.groupby('E2').P5.count() 
recent_data.groupby('E3').P5.mean() # E3이 1이면 비율이 낮다
recent_data.groupby('E3').P5.count() 
recent_data.groupby('E4').P5.mean() # E4 이용자 없음.
recent_data.groupby('E5').P5.mean() # 신판 이용자가 비율이 다소 높다.
recent_data.groupby('E5').P5.count() 
recent_data.groupby('E6').P5.mean() # "
recent_data.groupby('E6').P5.count() 

# 총 합 기준 분석.

pattern_count_data = pattern_data.astype(str)
shape = pattern_count_data.shape

for i in range(shape[0]):
    for j in range(shape[1]):
        pattern_count_data.iloc[i,j] = pattern_count_data.iloc[i,j].count('1') 
        
pattern_count_data

pattern_count_data.groupby('E1').P5.mean().plot()
pattern_count_data.groupby('E1').P5.count() # 한번도 안한사람이 더 확률이 높다

pattern_count_data.groupby('E2').P5.mean().plot()
pattern_count_data.groupby('E2').P5.count() # 음의 관계.

pattern_count_data.groupby('E3').P5.mean().plot()
pattern_count_data.groupby('E3').P5.count() # 음의 관계.

pattern_count_data.groupby('E5').P5.mean().plot()
pattern_count_data.groupby('E5').P5.count() 

pattern_count_data.groupby('E6').P5.mean().plot()
pattern_count_data.groupby('E6').P5.count() 


#패턴지수는 어케 도출하지?
## 가중이동평균이 적절할듯한데

decay = 0.9
def moveaverage(x, decay):
    x = list(map(int, str(x)))
    sum = 0
    for i in range(len(x)):
        sum += decay**(len(x)-i-1)*x[i]
    return sum

moveaverage('0010101', decay)

pattern_ma = pattern_data.applymap(lambda x : moveaverage(x, decay))

X = pattern_ma.iloc[:,:-1]
X = sm.add_constant(X)
X = X.drop(['E4'], axis = 1)
pattern_logreg = sm.GLM(y, X, family=sm.families.Binomial()).fit()
pattern_logreg.summary()
fpr, tpr, thresholds = roc_curve(y, pattern_logreg.predict(X))



# 일반 모형과 비교
X_con = pd.concat([recent_data, pattern_count_data], axis = 1).drop('P5', axis = 1)
X_con = sm.add_constant(X_con).astype(float)
pattern_con = sm.GLM(y, X_con, family = sm.families.Binomial()).fit()
fpr1, tpr1, thresholds1 = roc_curve(y, pattern_con.predict(X_con))
plt.plot(fpr1, tpr1)
## 거의 동일한 roc curve

# np.mean((pattern_con.predict(X_con) > 0.1) == y)
# np.mean((pattern_logreg.predict(X) > 0.1) == y)

# decays = [0.5, 0.6,0.7,0.8,0.9, 1]

# pattern_test = pattern_data[:100000] ; pattern_train = pattern_data[100000:]
# y_test = y[:100000] ; y_train = y[100000:]

# for i in decays:
#     pattern_ma = pattern_data.applymap(lambda x : moveaverage(x, i))
#     X_train = pattern_ma.iloc[:,:-1][100000:]; 
#     X_test = pattern_ma.iloc[:,:-1][:100000]
#     X_train = sm.add_constant(X_train)
#     X_test = sm.add_constant(X_test)
#     X_train = X_train.drop(['E4'], axis = 1)
#     X_test = X_test.drop(['E4'], axis = 1)

#     pattern_logreg = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
#     fpr, tpr, thresholds = roc_curve(y_test, pattern_logreg.predict(X_test))
#     plt.plot(fpr, tpr, label = i)
#     plt.legend()


    
# 랜덤포레스트 결정트리가 아주 좋은 결과를 낸다. <- 개소리 



## 패턴분석은 진짜 존나모르겠다.. 희소데이터는 pca를 돌릴필요가 있다.

from sklearn.decomposition import PCA
pattern_ma = pattern_data.applymap(lambda x : moveaverage(x, decay))
pca = PCA(n_components = 3)
pca_ma = pca.fit_transform(pattern_ma)
pca.explained_variance_ratio_
pca_log = LogisticRegression().fit(pca_ma, y)

def plot_roc(model, y, x):
    fpr, tpr, thresholds = roc_curve(y, model.predict(x))
    plt.plot(fpr, tpr)

# pca도 효과가 없네

import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Dense(10, input_shape = [13], activation = 'relu'),
    keras.layers.Dense(10, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
    ])

model.compile(loss = 'binary_crossentropy')
model.fit(X_con[100000:], y[100000:], epochs = 10)


#딥러닝도 별소용 없는거보니 이게 최선인가 보다.

#기본정보

personal_data
