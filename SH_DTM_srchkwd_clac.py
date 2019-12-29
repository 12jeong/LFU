# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:50:59 2019

@author: shyun
"""

#%%
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

# import sys
import pandas as pd
import seaborn as sns # 시각화
import matplotlib.pyplot as plt
# 그래프의 스타일을 지정
plt.style.use('ggplot') 
import scipy as sp
import sklearn
import matplotlib as mpl
import seaborn as sns
mpl.rcParams.update({'font.size':14})
plt.rc('font',family='Malgun Gothic') # windows
%matplotlib inline 

#%% expand output display pd.df
pd.set_option('display.expand_frame_repr', False) 

#%% Load raw data
raw_df_buy       = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\df_buy.csv",index_col=0) 
raw_trans_info   = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\trans_info.csv")
raw_online_bh    = pd.read_csv(pc+"Dropbox\\LFY\\datasets\\rawdata\\online_bh.csv")

df_buy      = raw_df_buy.copy()  
trans_info  = raw_trans_info.copy()
online_bh   = raw_online_bh.copy()

#%% 
df_buy      = raw_df_buy.copy()  
col_order = ['clnt_id', 'sess_dt','sess_id', 'hit_seq',# 분류 변수 
              'sech_kwd','action_type', # 아웃풋
             'hit_tm',  'biz_unit',  'hit_pss_tm', 'trans_id', 'tot_pag_view_ct','tot_sess_hr_v', 'trfc_src', 'dvc_ctg_nm', 'clnt_gender', 'clnt_age', # 기타 변수
             'id', 'buy'] # 추가 보조 변수
df_buy=df_buy[col_order]

df_buy = df_buy.sort_values(['clnt_id', 'sess_dt','sess_id', 'hit_seq','hit_tm'],axis=0) # 오름차순

# %% df_buy['trans_id'] backward fill ! : last_action !=6 인 경우 제거 : 'clnt_id','sess_dt','sess_id','hit_seq'별로 정렬 후 6 이후의 행 제거
df_buy['trans_id'] = df_buy.groupby(['clnt_id','sess_dt','sess_id'])['trans_id'].bfill()

#%% 키워드 리스트 - df_kwd ['kwdlist'] - buy_df
df_kwd = df_buy[df_buy['action_type']==0].groupby(['clnt_id','sess_dt','sess_id','trans_id'])['sech_kwd'].apply(list).reset_index()
df_kwd.columns = ['clnt_id', 'sess_dt', 'sess_id', 'trans_id', 'kwd_list']

#%% 대분류 리스트 - 
trans_info[['clnt_id', 'de_dt','trans_id']].drop_duplicates().shape #115110
df_clac1 = trans_info.groupby(['clnt_id', 'de_dt','trans_id'])['clac_nm1'].apply(list).reset_index()
df_clac2 = trans_info.groupby(['clnt_id', 'de_dt','trans_id'])['clac_nm2'].apply(list).reset_index()['clac_nm2']
df_clac3 = trans_info.groupby(['clnt_id', 'de_dt','trans_id'])['clac_nm3'].apply(list).reset_index()['clac_nm3']
df_clac = pd.concat([df_clac1, df_clac2,df_clac3], axis=1)
