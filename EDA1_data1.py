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

#%% Load raw data
data1 = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\제6회 L.POINT Big Data Competition\\제6회 L.POINT Big Data Competition-분석용데이터-01.온라인 행동 정보.csv",low_memory=False) 
data2 = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\제6회 L.POINT Big Data Competition\\제6회 L.POINT Big Data Competition-분석용데이터-02.거래 정보.csv") 
data3 = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\제6회 L.POINT Big Data Competition\\제6회 L.POINT Big Data Competition-분석용데이터-03.고객 Demographic 정보.csv") 
data4 = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\제6회 L.POINT Big Data Competition\\제6회 L.POINT Big Data Competition-분석용데이터-04.상품분류 정보.csv") 

#%%
#data1['clnt_id'] = data1['clnt_id'].astype(object)
#data1['trans_id'] = data1['trans_id'].astype(object)
#data1['sess_id'] = data1['sess_id'].astype(object)

data1.head()
data1["id"] = data1.index
data1["hit_pss_tm"] = data1["hit_pss_tm"]/(60*1000)

# action_type = 0인데 kwd 없는 경우는 없음.
sum(pd.isnull(data1[data1.action_type == 0].sech_kwd)) 

# 같은고객이 같은날짜에 같은 sess_id를 받는 경우는 없다.
d1_tmp1 = data1[['clnt_id','sess_id','sess_dt']]
d1_tmp2 = d1_tmp1.drop_duplicates() # unique한 정보만 모음
#d11 = data1[['clnt_id','sess_id','sess_dt','biz_unit']]
#d22 = np.where(d11.drop_duplicates().groupby(['clnt_id','sess_dt','sess_id',])['biz_unit'].agg('count').reset_index()['biz_unit']>1)
#d22.head()

# 같은고객이 다른날짜에 같은 sess_id를 받는 경우가 있을까? (YES)
d1_tmp3 = d1_tmp2.groupby(['clnt_id','sess_id'])['sess_dt'].agg('count').reset_index() 
d1_tmp3[d1_tmp3['sess_dt']>1]
data1[(data1['clnt_id']==2) & (data1['sess_id']==1)][['clnt_id','sess_id','sess_dt']].drop_duplicates() # 예시
# 다른고객, 같은날짜에 같은 sess_id를 받는 경우가 있을까? (YES)
d1_tmp4 = d1_tmp2.groupby(['sess_dt','sess_id'])['clnt_id'].agg('count').reset_index() 
d1_tmp4[d1_tmp4['clnt_id']>1]
data1[(data1['sess_dt']==20190930) & (data1['sess_id']==271)][['clnt_id','sess_id','sess_dt']].drop_duplicates() # 예시
# 따라서 group by에 대한 key는 clnt_id, sess_id, sess_dt 가 될 수 있음.


#%% 거래경험에 따른 고객 분류 
trans_T_key = data1[(np.isnan(data1.trans_id) == False) & (data1.action_type == 6 )][['clnt_id','sess_id','sess_dt']] # trans_id 가 존재하고 구매를 함
data1_trans_T = pd.merge(data1, trans_T_key, how='inner')
data1_trans_F = data1[~data1['id'].isin(data1_trans_T['id'])]

# 접속 시간에 따른 차이
hit_pss_tm_T = data1_trans_T['hit_pss_tm'] # 구매까지 걸린 접속시간
plt.hist(hit_pss_tm_T) 
plt.hist(hit_pss_tm_T[hit_pss_tm_T<100]) 
plt.boxplot(hit_pss_tm_T, showfliers=False) 

hit_pss_tm_F =  data1_trans_F.groupby(['clnt_id','sess_id','sess_dt'])['hit_pss_tm'].agg('max').reset_index()['hit_pss_tm'] # 접속 종료까지의 시간
plt.hist(hit_pss_tm_F)
plt.hist(hit_pss_tm_F[hit_pss_tm_F< 100]) # 분
plt.boxplot(hit_pss_tm_F, showfliers=False)

# 검색키워드가 있는지 차이
pd.isnull(data1_trans_T['sech_kwd']).sum() / data1_trans_T.shape[0] # 구매를 한 사람중에 키워드가 있는 비율
d1_tmp5 = data1_trans_F.groupby(['clnt_id','sess_id','sess_dt'],as_index=False).nth(0) 
pd.isnull(d1_tmp5['sech_kwd']).sum() / d1_tmp5.shape[0] # 구매를 안 한 사람중에 키워드가 있는 비율 # 이거 다시해야함

# 범주데이터
T_by_biz = data1_trans_T.groupby('biz_unit')['id'].agg('count')
T_by_trfc = data1_trans_T.groupby('trfc_src')['id'].agg('count')
T_by_dvc = data1_trans_T.groupby('dvc_ctg_nm')['id'].agg('count')

label = T_by_biz.index; index = np.arange(len(label)) ;plt.bar(index, T_by_biz) ; plt.xticks(index, label, fontsize=15)
label = T_by_trfc.index; index = np.arange(len(label)) ;plt.bar(index, T_by_trfc) ; plt.xticks(index, label, fontsize=15)
label = T_by_dvc.index; index = np.arange(len(label)) ;plt.bar(index, T_by_dvc) ; plt.xticks(index, label, fontsize=15)

d1_tmp_F = data1_trans_F.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'],as_index=False)
F_for_category = d1_tmp_F.nth(-1)

F_by_biz = F_for_category.groupby('biz_unit')['id'].agg('count')
F_by_trfc= F_for_category.groupby('trfc_src')['id'].agg('count')
F_by_dvc = F_for_category.groupby('dvc_ctg_nm')['id'].agg('count')

label = F_by_biz.index; index = np.arange(len(label)) ;plt.bar(index, F_by_biz) ; plt.xticks(index, label, fontsize=15)
label = F_by_trfc.index; index = np.arange(len(label)) ;plt.bar(index, F_by_trfc) ; plt.xticks(index, label, fontsize=15)
label = F_by_dvc.index; index = np.arange(len(label)) ;plt.bar(index, F_by_dvc) ; plt.xticks(index, label, fontsize=15)






d1_tmp5 = data1_trans_F[['clnt_id','sess_id','sess_dt']].drop_duplicates() 
pd.merge(data1_trans_F, d1_tmp5, how="right", on=['clnt_id','sess_id','sess_dt'])

d1_tmp6 = data1_trans_F[data1_trans_F['action_type']==0]
d1_tmp6.groupby(['clnt_id','sess_id','sess_dt']).nth(0) 



# help : ctrl+I or 함수()
# 검색키워드가있는 / 그룹별로 / 구매까지 걸린 시간 봐도 ㄱㅊ을듯

df = pd.DataFrame({"A":["foo", "foo", "foo", "bar"], "B":[0,1,1,1], "C":["A","A","B","A"]})
df
df.drop_duplicates(subset=['A', 'C'], keep=False)

# clnt_id, see_id, sess_dt 가 일치하는 자료만 뽑는다



data1_trans_T.info()

# session 접속 시간이 긴데 구매를 하는 고객과 구매를 하지 않는 고객의 차이 



data1_trans_T["clnt_id"]
# 거래를 한 사람중에 kwd가 존재하는 사람 (같은 sess_id 내에서 찾아야함)


