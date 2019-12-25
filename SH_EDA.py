%reset -f
#%% import
import os
os.getcwd()
from os import chdir
pc12 = "C:\\Users\\UOS\\"
pcsh = "C:\\Users\\user\\"
# pcyt = ""
pc = pcsh

os.chdir(pc+'Documents\\GITHUB\LFU')
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
raw_online_bh  = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\online_bh.csv")   #  3196362
raw_trans_info = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\trans_info.csv") 
raw_demo_info  = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\demo_info.csv") 
raw_prod_info  = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\prod_info.csv") 

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
prod_info_uk['pd_c'] = prod_info_uk['pd_c'].astype(str).str.zfill(4)
prod_info_uk.loc[1668]=['unknown','unknown','unknown','unknown']

trans_info = raw_trans_info.copy()
df_right = [raw_demo_info,prod_info_uk]
key = ['clnt_id','pd_c']
how = ['left','left']
for dtidx in range(len(df_right)):
    dt_temp = df_right[dtidx].copy().drop_duplicates()
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

#%%  nobuy 정의 : 동일 session내에서 action_type = 6(구매 완료) 없는 사람 

df_buy =online_bh.copy()
df_buy = df_buy[df_buy['action_type']==6]
df_buy.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt']).head()

buy_session_key = df_buy[['clnt_id','sess_id','sess_dt']]
buy_session_key['buy'] = 1
buy_session_key

temp = online_bh.copy()
temp=temp.merge(buy_session_key, how='left').drop_duplicates()

df_nobuy =temp[temp['buy'].isna()]
df_buy = temp[temp['buy'].notnull()]

df_nobuy.shape[0]/online_bh.shape[0] # 0.663
df_buy.shape[0]/online_bh.shape[0] # 0.336
 
#%%
raw_df_buy  = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\df_buy.csv") 
raw_df_nobuy  = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\df_no_buy.csv") 
df_nobuy.shape[0]/online_bh.shape[0] # 0.663
df_buy.shape[0]/online_bh.shape[0] # 0.336

#%%  refund 정의 : action_type == 7 이 존재하는 사람 
df_refund = online_bh.copy()
df_refund = df_refund[df_refund['action_type']==7]

refund_exp_clnt = df_refund['clnt_id']
refund_session_key = df_refund[['clnt_id','sess_id','sess_dt']]
refund_session_key['refund']=1
refund_session_key

temp = online_bh.copy()
temp=temp.merge(refund_session_key, how='left').drop_duplicates()

df_refund = temp[temp['refund'].notnull()]
df_norefund =temp[temp['refund'].isna()]

df_refund.shape[0]/online_bh.shape[0] # 0.018 


df_refund.to_csv(pc+"Dropbox\\LFY\\datasets\\df_refund.csv", index=False)
df_norefund.to_csv(pc+"Dropbox\\LFY\\datasets\\df_norefund.csv", index=False)
trans_info.to_csv(pc+"Dropbox\\LFY\\datasets\\mg_trans_info.csv", index=False)
online_bh.to_csv(pc+"Dropbox\\LFY\\datasets\\mg_online_bh.csv", index=False)

#=========================================================================
#%%  Segment : df_nobuy / df_buy / df_refund
#=========================================================================
# - trans_info : data 2 & 3 & 4


# trans_info.columns



# - df_refund or refund_exp_clnt 
# - srch_kwd | df_buy, df_nobuy
aa = df_buy[df_buy['action_type']==0 ]['sech_kwd']
df_nobuy[['action_type','sech_kwd']]
df_buy['sech_kwd']
# - 파생 변수 만들기 <- 충성도













