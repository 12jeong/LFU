%reset -f
#%% import
import os
os.getcwd()
from os import chdir
os.chdir('C:\\Users\\UOS\\Documents\\GITHUB\\LFU')
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

import shelve
#import pickle # for save.image like R 

#%% expand output display pd.df
pd.set_option('display.expand_frame_repr', False) 

#%% Load raw data
df_buy  = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\datasets\\df_buy.csv",index_col=0) 
df_nobuy = pd.read_csv("C:\\Users\\UOS\\Dropbox\\LFY\\datasets\\df_no_buy.csv",index_col=0) 


#%% buy
df1 = df_buy
df1['hit_diff']=df1.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'])['hit_pss_tm'].diff(periods=-1)*-1
df1['hit_diff_ratio'] = df1['hit_diff']/ df1['tot_sess_hr_v'] 

df1.head()

def func_count(x):
    d = {}
    d['count_action_0'] = x[x.action_type==0]['id'].count()
    d['count_action_1'] = x[x.action_type==1]['id'].count()
    d['count_action_2'] = x[x.action_type==2]['id'].count()
    d['count_action_3'] = x[x.action_type==3]['id'].count()
    d['count_action_4'] = x[x.action_type==4]['id'].count()
    d['count_action_5'] = x[x.action_type==5]['id'].count()
    d['count_action_6'] = x[x.action_type==6]['id'].count()
    d['count_action_7'] = x[x.action_type==7]['id'].count()    
    d['count_action_8'] = x[x.action_type==8]['id'].count()    
    return pd.Series(d, index=['count_action_0','count_action_1','count_action_2','count_action_3','count_action_4',
                               'count_action_5','count_action_6','count_action_7','count_action_8'])
def func_time(x):
    d = {}
    d['time_action_0'] = x[x.action_type==0]['hit_diff_ratio'].sum()
    d['time_action_1'] = x[x.action_type==1]['hit_diff_ratio'].sum()
    d['time_action_2'] = x[x.action_type==2]['hit_diff_ratio'].sum()
    d['time_action_3'] = x[x.action_type==3]['hit_diff_ratio'].sum()
    d['time_action_4'] = x[x.action_type==4]['hit_diff_ratio'].sum()
    d['time_action_5'] = x[x.action_type==5]['hit_diff_ratio'].sum()
    d['time_action_6'] = x[x.action_type==6]['hit_diff_ratio'].sum()
    d['time_action_7'] = x[x.action_type==7]['hit_diff_ratio'].sum()    
    d['time_action_8'] = x[x.action_type==8]['hit_diff_ratio'].sum()    
    return pd.Series(d, index=['time_action_0','time_action_1','time_action_2','time_action_3','time_action_4',
                               'time_action_5','time_action_6','time_action_7','time_action_8'])

df2 = df1.groupby(['clnt_id','sess_id','sess_dt']).apply(func_count) # count accorinding to action_type
df3 = df1.groupby(['clnt_id','sess_id','sess_dt']).apply(func_time)  # time consuming by action_type
df4 = df1.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'])['hit_seq'].agg([('tot_seq','max')])  # total hit_seq
df5 = df1.drop_duplicates(['clnt_id','sess_id','sess_dt']).drop(['hit_seq','action_type','hit_tm','hit_pss_tm','trans_id','sech_kwd','hit_diff','hit_diff_ratio'], axis=1)

df2.head()
df2.head()
df3.head()
df4.head()
df5.head()

df_buy_if = pd.merge(df2,df3,on=['clnt_id','sess_id','sess_dt'], how='inner')
df_buy_if = df_buy_if.merge(df4,on=['clnt_id','sess_id','sess_dt'], how='inner')
df_buy_if = df_buy_if.merge(df5,on=['clnt_id','sess_id','sess_dt'], how='inner')


#%% nobuy
df_nobuy['buy']= 0 # for Y = 0,1 

df6 = df_nobuy
df6['hit_diff']=df6.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'])['hit_pss_tm'].diff(periods=-1)*-1
df6['hit_diff_ratio'] = df6['hit_diff']/ df6['tot_sess_hr_v'] 

df7 = df6.groupby(['clnt_id','sess_id','sess_dt']).apply(func_count) # count accorinding to action_type
df8 = df6.groupby(['clnt_id','sess_id','sess_dt']).apply(func_time)  # time consuming by action_type
df9 = df6.sort_values(['clnt_id','sess_id','sess_dt','hit_seq']).groupby(['clnt_id','sess_id','sess_dt'])['hit_seq'].agg([('tot_seq','max')])  # total hit_seq
df10 = df6.drop_duplicates(['clnt_id','sess_id','sess_dt']).drop(['hit_seq','action_type','hit_tm','hit_pss_tm','trans_id','sech_kwd','hit_diff','hit_diff_ratio'], axis=1)

df_if_tmp = pd.merge(df7,df8,on=['clnt_id','sess_id','sess_dt'], how='inner')
df_if_tmp = df_if_tmp.merge(df9,on=['clnt_id','sess_id','sess_dt'], how='inner')
df_if_tmp = df_if_tmp.merge(df10,on=['clnt_id','sess_id','sess_dt'], how='inner')

#%% final table for customer information
df_design_buy = pd.concat([df_buy_if, df_if_tmp], axis=0)

import os
for item in os.environ:
    
filename='\\shelve.out'
my_shelf = shelve.open(filename,'n')
