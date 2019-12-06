#%% import
%reset -f
import os
os.getcwd()
from os import chdir

# import sys
import pandas as pd
import seaborn as sns # 시각화
import matplotlib.pyplot as plt
# 그래프의 스타일을 지정
plt.style.use('ggplot')
import numpy as np
import scipy as sp
import sklearn
import matplotlib as mpl
mpl.rcParams.update({'font.size':14})
plt.rc('font',family='Malgun Gothic') # windows
%matplotlib inline 

pd.set_option('display.expand_frame_repr', False) # expand output display pd.df
#%%
data1 = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFU\\제6회 L.POINT Big Data Competition-분석용데이터-01.온라인 행동 정보.csv",low_memory=False) 
data2 = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFU\\제6회 L.POINT Big Data Competition-분석용데이터-02.거래 정보.csv") 
data3 = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFU\\제6회 L.POINT Big Data Competition-분석용데이터-03.고객 Demographic 정보.csv") 
data4 = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFU\\제6회 L.POINT Big Data Competition-분석용데이터-04.상품분류 정보.csv") 

#%% data4 : 상품소분류코드, 상품 대분류명, 중분류명, 소분류명
data4.head()
data4.shape
data4.info()
data4.describe(include='all')

data4['clac_nm1'].value_counts()

#plt.figure(figsize=(20,10))
#sns.countplot(data=data4, y="clac_nm1")
data4_Meats = data4.loc[data4["clac_nm1"]=="Meats"]
sns.countplot(data=data4_Meats, y="clac_nm2")


#%% data3 : 고객ID, 성별, 연령대
data3.head() # 별로 필요 없을 것 같다
data3.shape
data3.info()
data3.loc[data3["clnt_gender"]!="unknown"].shape 
# 꽤있넴...정보있는 고객은 12245/72399
# 성별, 연령을 사용하고 싶으면 clnt_id로 연결해야할
# 믿을 수 있는 성별,연령인가? 멤버십으로 연계된 정보인가?

#%% data2 : 고객ID, 거래ID, 거래일련번호(구매순서), 업종단위, 상품소분류코드(data4["pd_c"]), 구매일자, 시각, 금액, 수량
data2.head()
data2.shape
data2.info()
data2.describe(include='all') 

data2['clnt_id'] = data2['clnt_id'].astype(object) # ID는 factor로 넣어야하지 않을까?
data2['trans_id'] = data2['trans_id'].astype(object)
data2.describe(include='all')  # unique 고객  11284.0  

data2['biz_unit'].unique() # 온라인(A) or 오프라인(B)

df_clnt = data2.clnt_id.value_counts()
data2.loc[data2["clnt_id"]==37102] 
# trans_id와 de_dt, de_tm은 겹치는 정보일것. 평일 or 주말, 시간은 의미 있을듯
# 무슨 상품구매했는데 data4와 연계 필요함

#%% data1 : 온라인 행동 정보   trans_id(data3) 거래정보 연계 필요, action_type (행동유형)이 key point인듯 + sech_kwd (검색 키워드)
data1.head()
data1.shape      # big data ~
data1.info()
data1.describe(include='all')

data1['clnt_id'] = data1['clnt_id'].astype(object)
data1.describe(include='all') # unique 고객은 72399

data1['dvc_ctg_nm'].unique() # nan, 모바일 웹, PC, 모바일 앱
data1['biz_unit'].unique() # A01, A02, A03
data1.loc[data1["biz_unit"]=="A01"].shape
data1.loc[(data1["biz_unit"]=="A01")&(data1["dvc_ctg_nm"]=="mobile_app")].shape
data1.loc[data1["biz_unit"]=="A02"] # 다름 



# 홈페이지를 한번 들어가봐야할듯
data1.sample = data1.loc[(data1["clnt_id"]==61252) &(data1["sess_dt"]==20190725)]
data1.sample.head()
data1.sample["hit_seq"].describe() # 125개인데 max는 100이다?
data1.sample.sort_values(by=['hit_seq'], axis=0) # 오름차순 정렬

data1.sample.loc[data1.sample["sess_id"]==80]["hit_seq"].describe() # count=100, max=100
data1_sample2 = data1.sample.loc[data1.sample["sess_id"]==80]
data1_sample2.sort_values(by=["hit_seq"],axis=0) # hit_pss_tm = 0이 없는걸 보니 전날 부터 접속한것같기도하고...
data1.loc[(data1["clnt_id"]==61252) & (data1["sess_id"]==80)].sort_values(by=['hit_seq'],axis=0)  # 아닌감?

# hit_seq = 1인데 hit_pss_tm = 0  아닌경우는 무엇일까 
# hit_pss_tm = 0인데 hit_seq = 1 아닌경우는 무엇일까 
data1.loc[data1["hit_pss_tm"]==0] .shape
data1.loc[(data1["hit_pss_tm"]==0) & (data1["hit_seq"]==1)] .shape
data1.loc[data1["hit_seq"]==1].shape

data1.loc[data1["hit_pss_tm"]==0] # hit_pss_tm 사이 시간이 중요할까? 글쎄...

# trans_id가 있는 거래수 : 1.8 %
# 기본적으로 action_type = 6 또는 7일 때 부여됨. 오류 발생가능
data1[np.isnan(data1.trans_id)==False].shape # 56989
data1[np.isnan(data1.trans_id)==False].shape[0]/data1.shape[0] * 100

data1_trans_T = data1[np.isnan(data1.trans_id)==False]
sum(data1_trans_T.action_type < 6) # trans_id가 있는데 action type이 6,7이 아닌경우는 없으나
data1_action_T =  data1[data1.action_type >= 6]
sum(np.isnan(data1_action_T.trans_id))  # action_type이 6,7인데 trans_id가 없는 경우는 있음.
data1_action_T[np.isnan(data1_action_T.trans_id)]
# example1
data1_action_T[(np.isnan(data1_action_T.trans_id)) & (data1_action_T.clnt_id==25279)]
data1[(data1.clnt_id==25279)& (data1.sess_id==33)].sort_values(by='hit_seq') # 거래가 되었는데 결제등의 이유로 오류가 난 경우인듯?
# example2
data1_action_T[(np.isnan(data1_action_T.trans_id)) & (data1_action_T.clnt_id==46677)]
data1[(data1.clnt_id==46677)& (data1.sess_id==16)].sort_values(by='hit_seq') # 그럴 것 같음...


# sech_kwd : 검색을 하지 않았을 때 구매 패턴과 검색을 했을 때 구매 패턴이 다를 것 같음.
# trfc_src : 유입채널은 무슨 의미를 가질까...  PUSH 메시지가 중요할 것 같음. 직접접속하는 고객유형같은걸로 나눌 수 있지 않을까? 

# portal로 들어왔어도 sech_kwd가 존재하는가?
data1[data1.trfc_src=="PORTAL_1"] # YES!
data1[data1.trfc_src=="PORTAL_2"] # YES!
data1[data1.trfc_src=="PORTAL_3"] # YES!
