%reset -f
#%% import
import os
os.getcwd()
from os import chdir
os.chdir('C:\\Users\\UOS\\Documents\\GITHUB\LFU')
pdir = os.getcwd() ;print(pdir)

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

#%%
pd.set_option('display.expand_frame_repr', False) # expand output display pd.df

#%% Load raw data
raw_online_bh  = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\datasets\\online_bh.csv")   #  3196362
raw_trans_info = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\datasets\\trans_info.csv") 
raw_demo_info  = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\datasets\\demo_info.csv") 
raw_prod_info  = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\datasets\\prod_info.csv") 

#%% copy raw data
online_bh = raw_online_bh.copy()
trans_info= raw_trans_info.copy()
demo_info = raw_demo_info.copy()
prod_info = raw_prod_info.copy()

#%% data1 & data3 merge
df_right = [raw_demo_info]
for dtidx in range(len(df_right)):
    dt_temp = df_right[dtidx].copy()
    online_bh = online_bh.merge(dt_temp, on='clnt_id', how='left')

#%% 2,3,4 merge
prod_info_uk=raw_prod_info.copy()
prod_info_uk.loc[1668]=['unknown','unknown','unknown','unknown']

trans_info = raw_trans_info.copy()
df_right = [raw_demo_info,prod_info_uk]
key = ['clnt_id','pd_c']
how = ['left','outer']
for dtidx in range(len(df_right)):
    dt_temp = df_right[dtidx].copy()
    trans_info = trans_info.merge(dt_temp, on=key[dtidx], how=how[dtidx])

#%% data1 : online_bh 전처리 
online_bh.head()
online_bh["id"] = online_bh.index
online_bh["hit_pss_tm"] = online_bh["hit_pss_tm"]/(60*1000)   # 분으로 변환
online_bh["tot_sess_hr_v"] = online_bh["tot_sess_hr_v"]/60    # 분으로 변환

# action_type == 6 or 7 인데 trans_id == NaN 인 행 제거
sum(np.isnan(online_bh[(online_bh.action_type == 6) | (online_bh.action_type == 7) ].trans_id)) 
online_bh = online_bh[~( (np.isnan(online_bh.trans_id) == True) & ((online_bh.action_type==6)| (online_bh.action_type == 7)))]
sum(np.isnan(online_bh[(online_bh.action_type == 6) | (online_bh.action_type == 7) ].trans_id)) 

## action_type == 7 & trans_id 중복된 행, 첫 행만 보존
df_action7 =online_bh.copy()
df_action7 = df_action7[df_action7['action_type']==7]
df_action7.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt']).head()
df_action7_nd = df_action7.drop_duplicates(['trans_id'],keep="first")
df_action7_nd.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt']).head()
df_action_n7 =online_bh.copy()
df_action_n7 = df_action_n7[df_action_n7['action_type']!=7]
online_bh=pd.concat([df_action7_nd, df_action_n7]) 

#%% online_bh 에서 구매를 한 사람의 정보
trans_id_key_tmp = online_bh[ (online_bh.action_type == 6 )][['clnt_id','sess_id','sess_dt','trans_id']]
trans_id_key = trans_id_key_tmp[trans_id_key_tmp['trans_id'].isin(trans_info.trans_id)] # trans_info 에 trans_id 가 있는 정보만 활용
df_buy_tmp = pd.merge(online_bh, trans_id_key.drop(['trans_id'],axis=1), how='inner').drop_duplicates()
#df_buy_tmp2 = df_buy_tmp.groupby(['clnt_id','sess_id','sess_dt','trans_id'])['id'].agg(trans_count='count') > 1 # 같은 trans_id가 한개 이상이면 구매 후 환불한것
#tmp_key = df_buy_tmp2[df_buy_tmp2.trans_count==True].reset_index()[['clnt_id','sess_id','sess_dt','trans_id']]
#tmp_df1  = pd.merge(online_bh, tmp_key , how='inner').sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'])
#tmp_df1.head()
df_buy_tmp2 = df_buy_tmp.groupby(['clnt_id','sess_id','sess_dt','trans_id'])['id'].agg(trans_count='count') < 2  # 구매만 한 데이터
tmp_key = df_buy_tmp2[df_buy_tmp2.trans_count==True].reset_index()[['clnt_id','sess_id','sess_dt']]
df_buy_tmp3  = pd.merge(df_buy_tmp, tmp_key , how='inner').drop_duplicates()
df_buy = df_buy_tmp3
df_nobuy = online_bh[~ online_bh['id'].isin(df_buy['id'])] 

df_buy.shape[0] + df_nobuy.shape[0] 
online_bh.shape[0]

# sum(df_buy_tmp.groupby(['clnt_id','sess_id','sess_dt','trans_id'])['trans_id'].agg('count') == 2) # 구매를 하고 바로 취소를 한 경우?
# sum(df_buy_tmp.groupby(['clnt_id','sess_dt','trans_id'])['trans_id'].agg('count') == 2) # 구매를 하고 같은날 취소를 한 경우 
# sum(df_buy_tmp.groupby(['clnt_id','trans_id'])['trans_id'].agg('count') == 2) # 구매를 하고 다른날 취소를 한 경우 


