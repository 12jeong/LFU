%reset -f
#%% import
import os
os.getcwd()
from os import chdir
os.chdir('C:\\Users\\UOS')
#os.chdir('C:\\Users\\MYCOM')
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
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
#%% expand output display pd.df
pd.set_option('display.expand_frame_repr', False) 

#%% Load raw data
df_buy  = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\df_buy.csv") 
df_nobuy = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\df_nobuy.csv")
online_bh = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\online_bh.csv")
trans_info = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\trans_info.csv")
df_design_buy = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\df_design_buy.csv",index_col=0)
mg_ppdata = pd.read_csv(".\Dropbox\\LFY\\datasets\\ppdata\\mg_ppdata.csv")

online_bh['sess_dt'] =  pd.to_datetime(online_bh['sess_dt'])
trans_info['de_dt'] =  pd.to_datetime(trans_info['de_dt'])
#%%

online_clnt_input = online_bh.iloc[100]

def recommendation_with_past(online_clnt_input):
    dt_tmp = online_clnt_input['sess_dt'].day_name()
    if ((dt_tmp == 'Saturday') | (dt_tmp=='Sunday')):
        online_clnt_input['weekend']=1
    else:
        online_clnt_input['weekend']=0
    
    biz_unit = online_clnt_input.biz_unit
    weekend = online_clnt_input.weekend
    clnt_age = online_clnt_input.clnt_age
    clnt_gender = online_clnt_input.clnt_gender

    if (clnt_gender != "unknown"):
        past_trans_info = trans_info[trans_info.biz_unit == biz_unit]
        past_trans_info['clnt_age'] = past_trans_info['clnt_age'].astype('str')
        past_clac = past_trans_info[(past_trans_info.clnt_age==clnt_age)&(past_trans_info.weekend==weekend)&
                                    (past_trans_info.clnt_gender==clnt_gender)].clac_nm1.value_counts(normalize=True).mul(100).index
        past_top3 = past_clac[:3]
       
        
        time_diff = (trans_info['de_dt'] - online_clnt_input['sess_dt'])
        recently_trans_info = time_diff
        lst = [time_diff[x].days for x in range(len(time_diff))]
        lst < 3
        past_trans_info[(past_trans_info.clnt_age==clnt_age)&(past_trans_info.weekend==weekend)&
                        (past_trans_info.clnt_gender==clnt_gender)].clac_nm1.value_counts(normalize=True).mul(100).iloc[0:10].plot.barh(stacked=True)
        plt.show()
        return(past_top3)
        
a = recommendation_with_past(online_clnt_input)

